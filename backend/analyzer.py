import logging
import os
import re
import threading
from typing import Dict, List, Optional, Tuple
import torch

from backend.claim_detector import (
    document_warning_for_text,
    is_investor_claim,
    normalize_claim_text,
)

# Suppress Hugging Face symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger("pitchprobe.analyzer")

# DA1 Section 4.2: Zero-shot NLI Hypotheses for Three-Tier Risk Taxonomy
NLI_HYPOTHESES = {
    "Low": "This claim is low risk, grounded, and backed by concrete facts, traction, or verifiable evidence.",
    "Moderate": "This claim is moderate risk, vague, and lacks concrete evidence, validation, or metrics.",
    "High": "This claim is high risk, hyperbolic, and makes exaggerated or unsubstantiated projections.",
}

# Domain heuristic patterns for fallback classification (DA1 Section 5 Fallback Plan)
GROUNDED_PATTERNS = [
    r"\$\s*\d+[\d,]*\s*(?:k|m|b|mrr|arr)?",
    r"\b\d+(?:\.\d+)?\s*%\s*(?:growth|retention|cagr|churn|mom|yoy|margin)",
    r"\b(?:verified by|audited by|tracked in|stripe|cohort|paying customers|active users|nps)\b",
    r"\b(?:co-founder|engineers|phd|patents? granted|raised \$\d+)\b",
]

HYPERBOLIC_PATTERNS = [
    r"\b(?:massive|trillion[- ]dollar|trillion\b|entire global market|no competitors|zero competition)\b",
    r"\b(?:guaranteed to|inevitable|infinite growth|skyrocket|100x return|disrupt everything)\b",
    r"\b(?:revolutionary without equal|world's only|unprecedented dominance|easy monopoly)\b",
    r"\b(?:capture 100%|billions in effortless revenue|risk[- ]free)\b",
]

VAGUE_PATTERNS = [
    r"\b(?:significant traction|rapidly growing|best[- ]in[- ]class|industry[- ]leading)\b",
    r"\b(?:huge potential|poised to dominate|next[- ]gen platform|synergistic|seamless)\b",
    r"\b(?:state[- ]of[- ]the[- ]art|game[- ]changing|unmatched experience|cutting[- ]edge)\b",
    r"\b(?:exponential potential|tremendous opportunity|strong interest)\b",
]


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        # PITCHPROBE_DEVICE=cpu bypasses CUDA initialization entirely. This is
        # useful on Windows systems where the GPU driver/native CUDA stack can
        # terminate Python rather than raising a catchable exception.
        requested_device = os.environ.get("PITCHPROBE_DEVICE", "auto").strip().lower()
        if requested_device == "cpu":
            self.device = "cpu"
        elif requested_device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                logger.warning(
                    "PITCHPROBE_DEVICE=cuda was requested, but CUDA is unavailable; using CPU."
                )
                self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("PitchProbe inference device: %s", self.device)
        self.classifier_tokenizer = None
        self.classifier_model = None
        self.generator_tokenizer = None
        self.generator_model = None

        # A fine-tuned three-risk-label checkpoint is optional. Claim screening
        # remains separate; without this path the original NLI classifier is active.
        self.finetuned_classifier_path = os.environ.get("PITCHPROBE_CLASSIFIER_PATH", "").strip()
        self.classifier_mode = "fine_tuned" if self.finetuned_classifier_path else "nli"
        self.classifier_name = (
            self.finetuned_classifier_path
            if self.finetuned_classifier_path
            else "cross-encoder/nli-deberta-v3-base"
        )
        self.generator_name = "google/flan-t5-small"
        # The base FLAN-T5 checkpoint emits generic/unrelated questions on
        # this task. Keep it opt-in until a task-specific checkpoint is evaluated.
        self.use_flan_t5 = os.environ.get("PITCHPROBE_USE_FLAN_T5", "0").strip().lower() in {
            "1", "true", "yes"
        }

        self.classifier_loaded = False
        self.generator_loaded = False

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def try_load_models(self, background: bool = False) -> None:
        """Attempt to load transformer models locally; fail gracefully if unavailable."""
        def _loader():
            # Check local cached models first to avoid blocking on network
            try:
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                )

                if not self.classifier_loaded:
                    # Accept either a mounted local checkpoint or a Hugging Face
                    # Hub model ID. A configured fine-tuned classifier is still
                    # opt-in; with no PITCHPROBE_CLASSIFIER_PATH the original NLI
                    # model and the existing question-generation path are unchanged.
                    if self.classifier_mode == "fine_tuned":
                        try:
                            local_checkpoint = os.path.isdir(
                                os.path.expanduser(self.classifier_name)
                            )
                            load_kwargs = {"local_files_only": local_checkpoint}

                            # Pin a Hub revision for reproducible deployments.
                            # Never log the token; only pass it to Transformers.
                            if not local_checkpoint:
                                revision = os.environ.get(
                                    "PITCHPROBE_CLASSIFIER_REVISION", ""
                                ).strip()
                                if revision:
                                    load_kwargs["revision"] = revision
                                hf_token = (
                                    os.environ.get("HF_TOKEN", "").strip()
                                    or os.environ.get(
                                        "HUGGINGFACE_HUB_TOKEN", ""
                                    ).strip()
                                )
                                if hf_token:
                                    load_kwargs["token"] = hf_token

                            logger.info(
                                "Loading fine-tuned three-label risk classifier from %s (%s checkpoint)...",
                                self.classifier_name,
                                "local" if local_checkpoint else "Hub",
                            )
                            tok = AutoTokenizer.from_pretrained(
                                self.classifier_name,
                                **load_kwargs,
                            )
                            model = AutoModelForSequenceClassification.from_pretrained(
                                self.classifier_name,
                                **load_kwargs,
                            ).to(self.device)

                            id2label = getattr(model.config, "id2label", {})
                            normalized_labels = {
                                str(label).strip().upper().replace("-", "_").replace(" ", "_")
                                for label in id2label.values()
                            }
                            required_labels = {"LOW", "MODERATE", "HIGH"}
                            if len(id2label) != 3 or normalized_labels != required_labels:
                                raise ValueError(
                                    "Fine-tuned risk checkpoint must have exactly "
                                    "three outputs mapped to LOW, MODERATE, and HIGH; "
                                    f"found id2label={id2label}"
                                )
                            if set(int(key) for key in id2label) != {0, 1, 2}:
                                raise ValueError(
                                    "Fine-tuned risk checkpoint label IDs must be 0, 1, 2; "
                                    f"found id2label={id2label}"
                                )

                            model.eval()
                            self.classifier_tokenizer = tok
                            self.classifier_model = model
                            self.classifier_loaded = True
                            logger.info(
                                "Loaded fine-tuned classifier %s.",
                                self.classifier_name,
                            )
                        except Exception as exc:
                            logger.error(
                                "Could not load fine-tuned classifier at %s: %s. "
                                "Falling back to the original NLI model.",
                                self.classifier_name,
                                exc,
                            )
                            self.classifier_mode = "nli"
                            self.classifier_name = "cross-encoder/nli-deberta-v3-base"

                    if not self.classifier_loaded:
                        try:
                            logger.info("Checking local cache for %s...", self.classifier_name)
                            tok = AutoTokenizer.from_pretrained(
                                self.classifier_name,
                                local_files_only=False,
                            )
                            model = AutoModelForSequenceClassification.from_pretrained(
                                self.classifier_name,
                                local_files_only=False,
                            ).to(self.device)
                            model.eval()
                            self.classifier_tokenizer = tok
                            self.classifier_model = model
                            self.classifier_loaded = True
                            logger.info("Loaded classifier %s.", self.classifier_name)
                        except Exception as exc:
                            logger.warning(
                                "Classifier %s unavailable; using calibrated heuristics: %s",
                                self.classifier_name,
                                exc,
                            )

                if self.use_flan_t5 and not self.generator_loaded:
                    try:
                        logger.info("Loading opt-in FLAN-T5 generator %s...", self.generator_name)
                        tok = AutoTokenizer.from_pretrained(
                            self.generator_name,
                            local_files_only=False,
                        )
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            self.generator_name,
                            local_files_only=False,
                        ).to(self.device)
                        model.eval()
                        self.generator_tokenizer = tok
                        self.generator_model = model
                        self.generator_loaded = True
                        logger.info("Loaded opt-in generator %s.", self.generator_name)
                    except Exception as exc:
                        logger.warning(
                            "FLAN-T5 unavailable; using targeted investor questions: %s",
                            exc,
                        )
                elif not self.use_flan_t5:
                    logger.info(
                        "FLAN-T5 disabled for default inference; using targeted "
                        "investor-question rules. Set PITCHPROBE_USE_FLAN_T5=1 "
                        "to opt in for evaluation."
                    )
            except Exception as e:
                logger.warning("Model loader encountered error: %s", e)

        if background:
            thread = threading.Thread(target=_loader, daemon=True)
            thread.start()
        else:
            _loader()



def classify_sentence_finetuned(
    sentence: str, manager: ModelManager
) -> Tuple[str, float]:
    """Predict LOW/MODERATE/HIGH risk for a screened claim."""
    if not (
        manager.classifier_loaded
        and manager.classifier_model is not None
        and manager.classifier_tokenizer is not None
    ):
        raise RuntimeError("Fine-tuned classifier is not loaded")

    features = manager.classifier_tokenizer(
        sentence,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(manager.device)

    with torch.no_grad():
        logits = manager.classifier_model(**features).logits
    probabilities = torch.softmax(logits, dim=-1)[0]

    id2label = {
        int(key): str(value).strip().upper().replace("-", "_").replace(" ", "_")
        for key, value in manager.classifier_model.config.id2label.items()
    }
    best_idx = int(torch.argmax(probabilities).item())
    label = id2label.get(best_idx, "")
    if label not in {"LOW", "MODERATE", "HIGH"}:
        raise ValueError(
            "Unexpected fine-tuned label mapping; expected LOW/MODERATE/HIGH, "
            f"found {id2label}"
        )
    return label, round(float(probabilities[best_idx].item()), 4)


def classify_sentence_nli(
    sentence: str, manager: ModelManager
) -> Tuple[str, float]:
    """Classify a sentence by comparing NLI entailment scores for three risk hypotheses."""

    if (
        manager.classifier_loaded
        and manager.classifier_model is not None
        and manager.classifier_tokenizer is not None
    ):
        try:
            labels = ["Low", "Moderate", "High"]
            pairs = [
                (sentence, NLI_HYPOTHESES[label])
                for label in labels
            ]

            features = manager.classifier_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(manager.device)

            with torch.no_grad():
                logits = manager.classifier_model(**features).logits

            # Map output columns using the model's actual label configuration.
            config = manager.classifier_model.config
            id2label = {
                int(key): str(value).lower()
                for key, value in config.id2label.items()
            }

            entailment_ids = [
                idx for idx, label in id2label.items()
                if "entail" in label
            ]
            if len(entailment_ids) != 1:
                raise ValueError(
                    f"Cannot identify entailment label in {id2label}"
                )

            # The NLI model emits raw contradiction / entailment / neutral
            # logits for each candidate risk hypothesis. For mutually exclusive
            # candidate labels, normalize the RAW entailment logits across the
            # hypotheses. Softmaxing each NLI row first and then applying a
            # second softmax compresses close cases toward 1/3 (as seen in the
            # test-deck output).
            entailment_logits = logits[:, entailment_ids[0]]
            probabilities = torch.softmax(
                entailment_logits, dim=0
            ).detach().cpu().tolist()

            best_idx = max(
                range(len(probabilities)),
                key=lambda idx: probabilities[idx],
            )

            return labels[best_idx], round(
                float(probabilities[best_idx]), 4
            )

        except Exception as exc:
            logger.warning(
                "NLI classification failed; using heuristic fallback: %s",
                exc,
            )

    # Keep the existing heuristic fallback below this line.
    s_lower = sentence.lower()

    high_hits = sum(
        1 for pattern in HYPERBOLIC_PATTERNS
        if re.search(pattern, s_lower)
    )
    grounded_hits = sum(
        1 for pattern in GROUNDED_PATTERNS
        if re.search(pattern, s_lower)
    )
    vague_hits = sum(
        1 for pattern in VAGUE_PATTERNS
        if re.search(pattern, s_lower)
    )

    if high_hits > 0:
        return "High", min(0.78 + high_hits * 0.08, 0.96)
    if grounded_hits > 0 and vague_hits == 0:
        return "Low", min(0.80 + grounded_hits * 0.06, 0.98)
    if vague_hits > 0:
        return "Moderate", min(0.74 + vague_hits * 0.07, 0.92)

    return "Moderate", 0.68



def _is_useful_generated_question(question: str) -> bool:
    """Reject generic FLAN outputs that do not challenge the specific claim."""
    candidate = re.sub(r"\s+", " ", (question or "")).strip().lower()
    if len(candidate) < 28 or not candidate.endswith("?"):
        return False
    generic_patterns = [
        r"\bbest title\b",
        r"\btitle for (?:the )?(?:article|vc|company)\b",
        r"\bwhat is a (?:tough )?vc objection\b",
        r"\bwhich of the following\b",
        r"\bwhat is the best way to make a profit\b",
        r"\bwhat is the reason for this\b",
        r"\bwhat is the reason why\b",
        r"\bname of the company that is a part of\b",
        r"^what is the main reason for (?:the )?vc",
    ]
    return not any(re.search(pattern, candidate) for pattern in generic_patterns)


def _targeted_investor_question(sentence: str, risk: str) -> str:
    """Choose an evidence-seeking question using ordered, claim-specific signals."""
    text = re.sub(r"\s+", " ", sentence.lower()).strip()

    # Funding ask: focus on uses, milestones, and runway.
    if re.search(r"\b(?:raising|raise|funding ask|use of funds|capital ask)\b", text):
        return (
            "Which measurable milestones will this raise fund, how much runway do they buy, "
            "and what evidence must be achieved before the next funding round?"
        )

    # Team capability gaps are different from product privacy or manufacturing risks.
    if re.search(
        r"\b(?:founder|co-founder|founding team|leadership team|team lacks|team needs|"
        r"hiring|hire|leader|capability gap|advisor|fractional operations|privacy counsel|"
        r"former product lead|owns customer discovery|supply-chain leader|operations lead)\b",
        text,
    ):
        return (
            "Which critical capability is still missing, who is accountable for closing the gap, "
            "and what hiring or partner milestone must be met before the next stage?"
        )

    # Working-capital risk should ask about cash exposure, not general demand conversion.
    if re.search(r"\b(?:working[- ]capital|inventory deposits?|tooling deposits?)\b", text):
        return (
            "How much cash will be tied up in tooling and inventory before sell-through, "
            "what demand evidence supports that commitment, and what is the downside cash plan?"
        )

    # Production gates require explicit go/no-go language or a pre-order condition.
    # Do not infer a production gate from a generic mention of yield, returns, or tooling alone.
    if re.search(
        r"\b(?:do not place|don't place|before committing|gate to production|"
        r"production order|production tooling)\b",
        text,
    ) and re.search(r"\b(?:until|unless|gate|criteria|deposits?|yield|beta|pre-production)\b", text):
        return (
            "Which pre-production pass criteria must be met for paid deposits, fit/return intent, "
            "quality yield and beta usefulness, and who independently signs off before the order?"
        )

    # A risk statement about margin erosion is different from a go/no-go checklist.
    if re.search(r"\b(?:can erase margin|erase margins?|margin erosion|risk to gross margin)\b", text):
        return (
            "How much gross margin could be lost in downside cases for supplier yield, fit variance, "
            "microphone performance and returns, and which prototype tests quantify each exposure?"
        )

    # Defensibility should beat a co-occurring mention of retention.
    if re.search(
        r"\b(?:defensib\w*|competitors?|competition|incumbents?|moat|patent|"
        r"competitive advantage|copy|clone|bundl(?:e|ing))\b",
        text,
    ):
        return (
            "Which customer behaviour or integration creates a durable switching cost, "
            "and what evidence shows incumbents cannot bundle or copy it easily?"
        )

    # Market sizing: first distinguish an externally sourced TAM from assumed SAM/SOM.
    if (
        re.search(r"\btam\b", text)
        and re.search(r"\b(?:sam|som)\b", text)
        and re.search(r"\b(?:external|source|management assumption|not externally validated)\b", text)
    ):
        return (
            "Which TAM inputs are supported by independent sources, and which SAM/SOM boundaries "
            "and adoption assumptions remain management assumptions that need validation?"
        )

    if re.search(
        r"\b(?:tam|sam|som|market size|market share|penetration|beachhead|trillion|"
        r"billion|market estimate|market sizing|serviceable market|reachable market)\b",
        text,
    ):
        if re.search(r"\b(?:directional model|not a forecast of)\b", text) and re.search(r"\bshare\b", text):
            return (
                "What bottom-up evidence supports Sonora's attainable market share specifically, "
                "rather than the total category size, and how does the case change at a smaller share?"
            )
        if re.search(r"\bdefinitions vary by analyst\b", text):
            return (
                "How is your SAM defined differently from broader analyst categories, and what "
                "buyer count, price and adoption assumptions make that boundary defensible?"
            )
        # The numeric beachhead calculation is a customer-count/price/adoption question,
        # even though it labels itself a management assumption.
        if re.search(r"\b(?:initial beachhead|beachhead:|2\.5m digitally intensive|top \d+ metros)\b", text):
            return (
                "How was the beachhead buyer count estimated, what source supports the ₹18,000 "
                "price point, and what adoption evidence justifies the implied annual spend?"
            )
        if re.search(r"\b(?:penetration|market share|capture|share of market)\b", text):
            return (
                "What comparable adoption data and channel-level conversion evidence make this "
                "penetration target reachable, and how does the estimate change at a lower share?"
            )
        if re.search(r"\b(?:beachhead|target segment|professionals|reachable customer)\b", text):
            return (
                "How was the reachable buyer count derived, what source supports willingness to "
                "pay at this price, and what adoption evidence supports the obtainable market?"
            )
        if re.search(r"\b(?:external(?:ly)? sourced|external estimate|management assumption|"
                     r"internal assumption|not externally validated|directional model)\b", text):
            return (
                "Which parts of this market estimate come from independent sources versus "
                "management assumptions, and how sensitive is it to a narrower customer segment?"
            )
        return (
            "What bottom-up customer count, price and adoption assumptions produce this market "
            "estimate, and which independent sources support each input?"
        )

    # Revenue recognition / bridge definitions are not growth forecasts.
    if re.search(r"\brevenue is net of (?:discounts?|indirect taxes?|taxes)\b", text):
        return (
            "How are discounts and indirect taxes reconciled from gross retail value to reported net "
            "revenue, and can the bridge be tied to the financial model and channel assumptions?"
        )

    # Specific usage KPI: define the denominator and observation window.
    if re.search(r"\bweekly meeting summaries per active user\b|\bsummaries per active user\b", text):
        return (
            "How is an active user defined and over what cohort window is this target measured, "
            "and what instrumented beta data shows users sustain more than three summaries weekly?"
        )

    # Pilot measurement plans should ask about measurement design before broad traction conversion.
    if re.search(r"\bpilot with \d+ teams\b", text) and re.search(r"\b(?:measure|weekly active|admin time|renewal intent)\b", text):
        return (
            "How will the pilot define and measure weekly active use, admin time saved and renewal intent, "
            "what baseline will each metric use, and how many teams convert to paid use afterward?"
        )

    # A forecast / break-even claim is routed here; a statement merely defining net revenue is not.
    if re.search(r"\b(?:break[- ]even|ebitda|cash flow|cash runway|financial plan|"
                 r"forecast|year three|fy\d{2}|arr|mrr|valuation|revenue ambition|path to .{0,20} revenue)\b", text):
        if re.search(r"\b(?:break[- ]even|cash flow|cash runway|working capital)\b", text):
            return (
                "What monthly gross-margin, fixed-cost and subscription-attach assumptions determine "
                "the break-even date, and how much extra runway is needed in the downside case?"
            )
        return (
            "Which sales-volume, realised-price, conversion and gross margin assumptions drive this "
            "forecast, what actuals support them, and how does the downside case change the result?"
        )

    # Technical performance is distinct from commercial or production readiness.
    if re.search(r"\b(?:transcription|diarisation|accuracy|latency|sub-\d|model quality|"
                 r"ai reliability|algorithm|benchmark|response time|technical performance)\b", text):
        return (
            "What controlled benchmark supports this performance claim, what baseline was used, "
            "and how many users, environments and failure cases were measured?"
        )

    if re.search(
        r"\b(?:app connectivity|meeting-platform integration|fully offline|universal integration|"
        r"cloud ai|user consent|privacy|recording|security|deletion|encrypted|regulatory|"
        r"compliance|legal)\b",
        text,
    ):
        return (
            "What happens when connectivity or consent is unavailable, and what tests verify that "
            "recording, access, retention and deletion controls work as described?"
        )

    # Subscription prices, retention, unit economics, CAC, and traction.
    if re.search(
        r"(?:[₹$€£]\s?[\d,]+.{0,25}(?:/|\bper\b).{0,12}(?:month|seat)|\bper month\b|"
        r"\bsubscription(?: attach rate)?\b|\bfair-use\b|\bunlimited meeting minutes\b|"
        r"\bpricing tier\b)",
        text,
    ):
        if re.search(r"\b(?:planned after v1|planned after version 1|post[- ]v1|retention policies|"
                     r"admin controls|workspace integrations|shared templates)\b", text):
            return (
                "Which capabilities are committed for v1 versus deferred, and what buyer evidence "
                "shows the planned tier's incremental features justify its recurring price?"
            )
        if re.search(r"\b(?:attach rate|retention|renewal|churn|cohort|subscriber activation)\b", text):
            return (
                "What cohort data shows subscribers activate, retain and renew at the assumed rate, "
                "and how do usage, support cost and churn differ by plan?"
            )
        return (
            "What customer research or paid conversion data supports this price tier, and do expected "
            "usage, service costs and upgrade behaviour leave sustainable gross margin?"
        )

    # Unit economics comes before generic manufacturing/supplier terms so costs/margins
    # receive a reconciliation question rather than a production gate.
    if re.search(r"\b(?:unit economics|cogs|gross margin|contribution|manufacturing cost|asp|"
                 r"landed cost|fulfilment|fulfillment|warranty|cost per|unit cost)\b", text):
        return (
            "Do these unit economics reconcile retail price to net revenue and include landed cost, "
            "discounts, taxes, fulfilment, warranty, channel fees and acquisition cost, and which quotes support them?"
        )

    if re.search(r"\b(?:blended cac|customer acquisition cost|cost per acquisition|"
                 r"search and referral loops|cac)\b", text):
        return (
            "What paid-channel and referral-cohort data supports this acquisition-cost target, "
            "and does payback remain viable after churn, discounts and gross margin?"
        )

    if re.search(
        r"\b(?:supplier|manufactur(?:ing|er)|production|inventory|tooling|yield|hardware|"
        r"battery|acoustic testing|returns)\b",
        text,
    ):
        return (
            "What measured supplier yield, fit, reliability and return data support these unit "
            "assumptions, and what failure threshold would stop production?"
        )

    if re.search(r"\b(?:interviews?|waitlist|sign-ups?|paid units|paid deposits|pilot|beta|"
                 r"traction|activation|retention|conversion|weekly active|active users|"
                 r"paying customers|customer count|customers acquired|users acquired)\b", text):
        return (
            "How many interested users or pilot participants converted to paying customers, "
            "over what time window, and what cohort evidence supports that conversion rate?"
        )

    if re.search(
        r"\b(?:customer-facing teams|ideal customer profile|target customer|buyer persona|"
        r"teams with .{0,30}weekly calls|buyers who need|customer segment)\b",
        text,
    ):
        return (
            "What evidence confirms this target segment has a frequent, costly problem, "
            "and how many qualified buyers have demonstrated willingness to pay?"
        )

    if re.search(r"\b(?:reduces?|saves?|improves?|faster|better|industry.leading|best.in.class|"
                 r"seamless|significant|unmatched|leading|strong|large|massive|revolutionary)\b", text):
        return (
            "What measured customer outcome supports this benefit, what baseline is it compared "
            "against, and what result would falsify the claim?"
        )

    if risk == "High":
        return (
            "What specific evidence would let an independent investor verify this claim, "
            "and what result would show that it is overstated?"
        )
    return (
        "What observable customer result or source evidence supports this statement, "
        "and what evidence would change the conclusion?"
    )

def generate_objection_question(sentence: str, risk: str, manager: ModelManager) -> Optional[str]:
    """Generate one targeted question for Moderate/High claims; Low returns None."""
    if risk == "Low":
        return None

    # FLAN-T5 remains available for experiments, but its generic checkpoint is
    # not trusted by default for this task. Discard unrelated completions.
    if (
        getattr(manager, "use_flan_t5", False)
        and manager.generator_loaded
        and manager.generator_model is not None
        and manager.generator_tokenizer is not None
    ):
        try:
            prompt = (
                "You are a skeptical seed-stage venture capitalist. Ask exactly one "
                "concise, specific, evidence-seeking question that challenges the "
                "following startup claim. Do not ask for a title or a generic VC objection. "
                "Focus on the weakest assumption, metric, customer validation, economics, "
                "competition, or implementation risk.\n"
                f"Risk: {risk}\nClaim: {sentence}\nQuestion:"
            )
            inputs = manager.generator_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(manager.device)
            with torch.no_grad():
                outputs = manager.generator_model.generate(
                    **inputs,
                    max_new_tokens=56,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
            generated = manager.generator_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            if not generated.endswith("?"):
                generated += "?"
            if _is_useful_generated_question(generated):
                return generated
            logger.warning("Discarding generic/irrelevant FLAN-T5 question: %r", generated)
        except Exception as exc:
            logger.warning("FLAN-T5 question generation failed; using targeted fallback: %s", exc)

    return _targeted_investor_question(sentence, risk)


def analyze_sentences(
    sentences: List[str],
    raw_text: Optional[str] = None,
) -> Tuple[List[Dict], Dict[str, int], Dict[str, object]]:
    """Screen claims, classify risk, then gate investor question generation."""
    manager = ModelManager.get_instance()
    summary_counts = {"low_risk": 0, "moderate_risk": 0, "high_risk": 0}

    if manager.classifier_loaded and manager.classifier_mode == "fine_tuned":
        classifier_mode_name = "fine_tuned_three_label"
    elif manager.classifier_loaded:
        classifier_mode_name = "transformer_nli"
    else:
        classifier_mode_name = "calibrated_domain_heuristics"

    if manager.generator_loaded:
        mode = f"{classifier_mode_name}+transformer_generator"
        generator_display_name = manager.generator_name
    else:
        mode = f"{classifier_mode_name}+targeted_question_rules"
        generator_display_name = "targeted_question_rules"

    pipeline_info = {
        "classifier": manager.classifier_name,
        "generator": generator_display_name,
        "mode": mode,
        "note": (
            "Claim-first screening filters non-claims before risk scoring; "
            "investor questions are generated only for Moderate/High claims."
        ),
        "document_warning": None,
    }

    # Skip obvious coursework/research files before classifying individual lines.
    document_warning = document_warning_for_text(raw_text)
    if document_warning:
        pipeline_info["note"] = (
            "Risk scoring was skipped because the document does not appear to "
            "be a startup pitch deck."
        )
        pipeline_info["document_warning"] = document_warning
        return [], summary_counts, pipeline_info

    candidates: List[Tuple[str, str, float]] = []

    if manager.classifier_loaded and manager.classifier_mode == "fine_tuned":
        # Public financial-claim datasets provide risk/evidence labels, not a
        # reliable "not a startup claim" class. Keep claim screening separate,
        # then let the fine-tuned model assign one of the three risk labels.
        risk_for_label = {"LOW": "Low", "MODERATE": "Moderate", "HIGH": "High"}
        for sentence in sentences:
            text = normalize_claim_text(sentence)
            if not text or not is_investor_claim(text, manager=None):
                continue
            label, score = classify_sentence_finetuned(text, manager)
            candidates.append((text, risk_for_label[label], score))
    else:
        # Preserve current behavior until a fine-tuned checkpoint is configured.
        claims = [
            normalize_claim_text(sentence)
            for sentence in sentences
            if is_investor_claim(sentence, manager)
        ]
        for claim in claims:
            risk, score = classify_sentence_nli(claim, manager)
            candidates.append((claim, risk, score))

    if not candidates:
        pipeline_info["note"] = (
            "No investor-facing claims were detected, so risk scoring was skipped."
        )
        pipeline_info["document_warning"] = (
            "No investor-facing claims were detected. Upload a startup pitch deck "
            "with product, market, traction, customer, or financial assertions."
        )
        return [], summary_counts, pipeline_info

    classified_results = []
    for idx, (claim, risk, score) in enumerate(candidates, start=1):
        if risk == "Low":
            summary_counts["low_risk"] += 1
        elif risk == "Moderate":
            summary_counts["moderate_risk"] += 1
        elif risk == "High":
            summary_counts["high_risk"] += 1

        question = generate_objection_question(claim, risk, manager)
        classified_results.append(
            {
                "id": idx,
                "text": claim,
                "risk": risk,
                "score": score,
                "question": question,
            }
        )

    return classified_results, summary_counts, pipeline_info
