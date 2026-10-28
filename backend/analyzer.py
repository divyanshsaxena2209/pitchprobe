import logging
import os
import random
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
                    # Fine-tuned checkpoints are local-only: do not silently
                    # download an unrelated model if the configured path is bad.
                    if self.classifier_mode == "fine_tuned":
                        try:
                            logger.info(
                                "Loading fine-tuned three-label risk classifier from %s...",
                                self.classifier_name,
                            )
                            tok = AutoTokenizer.from_pretrained(
                                self.classifier_name,
                                local_files_only=True,
                            )
                            model = AutoModelForSequenceClassification.from_pretrained(
                                self.classifier_name,
                                local_files_only=True,
                            ).to(self.device)
                            normalized_labels = {
                                str(label).strip().upper().replace("-", "_").replace(" ", "_")
                                for label in model.config.id2label.values()
                            }
                            required_labels = {"LOW", "MODERATE", "HIGH"}
                            if not required_labels.issubset(normalized_labels):
                                raise ValueError(
                                    "Fine-tuned risk checkpoint must define labels "
                                    "LOW, MODERATE, and HIGH; found "
                                    f"{sorted(normalized_labels)}"
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
    """Generate one evidence-seeking VC question based on the claim's main topic."""
    text = sentence.lower()

    # Strong domain-specific outcomes must not fall through to a generic
    # "what evidence supports this?" template.
    if re.search(
        r"\b(?:clinical|medical|therapeutic|knee pain|pain reduction|"
        r"reduces? pain|treats? symptoms?|diagnos(?:e|is)|injury reduction)\b",
        text,
    ):
        return (
            "What controlled study, comparison group, and pre-defined outcome measure "
            "substantiate this health or pain-reduction claim?"
        )

    # Physical consumer-product claims need representative trials, repeated-use
    # evidence, and a measured baseline.
    if re.search(
        r"\b(?:comfort|fit consistency|fit-related|durability|wear trial|wear test|"
        r"footwear|sneakers?|shoes?|insole|outsole|shoe weight|prototype durability|"
        r"return rate)\b",
        text,
    ):
        return (
            "What was the wear-test protocol, who participated, and how did the "
            "comfort, fit, or durability result compare with a measured baseline?"
        )

    # Audio hardware and meeting-intelligence claims need tests under realistic
    # environmental conditions, not just a controlled demo.
    if re.search(
        r"\b(?:anc|active noise cancellation|noise control|microphone|mic array|"
        r"voice pickup|battery life|battery runtime|earbuds?|earable|"
        r"speaker diarisation|diarisation|transcription accuracy|meeting capture)\b",
        text,
    ):
        return (
            "Under which real-world noise, call, or battery conditions was this tested, "
            "what benchmark was used, and how many trials were measured?"
        )

    # Fundraising asks should focus on milestone/use-of-funds discipline rather
    # than treating the ask as a revenue forecast.
    if re.search(r"\b(?:raising|raise|funding ask|use of funds|capital ask)\b", text):
        return (
            "Which measurable milestones will this raise fund, how much runway do they buy, "
            "and what evidence must be achieved before the next funding round?"
        )

    # Forecasts come before generic conversion/traction matches.
    if re.search(r"\b(revenue|forecast|financial plan|ebitda|runway|valuation|arr|mrr|year three|fy\d{2})\b", text):
        return (
            "Which assumptions for sales volume, realised price, conversion, and gross "
            "margin drive this forecast, and what evidence validates them?"
        )

    if re.search(r"\b(tam|sam|som|market size|market share|penetration|beachhead|trillion|billion|market estimate|market sizing)\b", text):
        return (
            "What bottom-up assumptions for the reachable customer segment, price, "
            "and adoption rate support this market estimate?"
        )

    # A customer profile is not proof of traction; ask about pain and willingness to pay.
    if re.search(r"\b(customer-facing teams|ideal customer profile|target customer|buyer persona|teams with .{0,30}weekly calls|buyers who need)\b", text):
        return (
            "What evidence confirms this target segment has a frequent, costly problem, "
            "and how many qualified buyers have shown willingness to pay?"
        )

    # Subscription/tier pricing is different from hardware unit economics.
    if re.search(r"(?:[₹$€£]\s?[\d,]+.{0,25}(?:/|\bper\b).{0,12}(?:month|seat)|\bper month\b|\bsubscription(?: attach rate)?\b|\bfair-use\b|\bunlimited meeting minutes\b|\bpricing tier\b)", text):
        if re.search(r"\b(?:attach rate|retention|renewal|churn|cohort)\b", text):
            return (
                "What cohort data shows that subscribers activate, retain and renew at the "
                "assumed rate, and how does usage vary by plan?"
            )
        return (
            "What customer research or paid conversion data supports this price tier, and "
            "do expected usage and service costs leave sustainable gross margin?"
        )

    if re.search(r"\b(unit economics|cogs|gross margin|contribution|manufacturing cost|asp|fulfilment|fulfillment|warranty|cost per|unit cost)\b", text):
        return (
            "Do these unit economics include discounts, returns, fulfilment, warranty, "
            "channel fees, and acquisition cost, and which supplier quotes support them?"
        )

    if re.search(r"\b(?:blended cac|customer acquisition cost|cost per acquisition|search and referral loops)\b", text):
        return (
            "What paid-channel and referral-cohort data supports this acquisition-cost target, "
            "and does the payback period remain viable after churn and gross margin?"
        )

    if re.search(r"\b(interviews?|waitlist|sign-ups?|paid units|paid deposits|pilot|beta|traction|activation|retention|conversion|weekly active|active users|paying customers|customer count|customers acquired|users acquired)\b", text):
        return (
            "How many interested users or pilot participants converted to paying customers, "
            "and what cohort evidence supports that conversion rate?"
        )

    # Team/role language is specific; don't match generic product copy such as "team templates".
    if re.search(r"\b(founder|co-founder|founding team|leadership team|team lacks|team needs|hiring|hire|leader|capability gap|advisor|fractional operations|privacy counsel|former product lead|owns customer discovery|supply-chain leader|operations lead)\b", text):
        return (
            "Which critical capability is currently missing, who will own it, and what "
            "is the hiring or partner timeline before the next milestone?"
        )

    if re.search(r"\b(competitors?|competition|incumbents?|moat|defensib|patent|competitive advantage|clone|copy)\b", text):
        return (
            "What prevents established competitors from bundling or copying this, and "
            "what evidence shows customers would switch to your product?"
        )

    if re.search(r"\b(transcription|diarisation|accuracy|latency|sub-\d|model quality|ai reliability|algorithm|benchmark|response time|technical performance)\b", text):
        return (
            "What controlled benchmark supports this performance claim, what baseline "
            "did you compare against, and how many users or trials were measured?"
        )

    if re.search(r"\b(app connectivity|meeting-platform integration|fully offline|universal integration|cloud ai|user consent|privacy|recording|security|deletion|encrypted|regulatory|compliance|legal)\b", text):
        return (
            "What happens when connectivity or consent is unavailable, and what tests verify "
            "that the product respects user permissions across supported platforms?"
        )

    if re.search(r"\b(supplier|manufactur|production|inventory|tooling|yield|hardware|battery|fit-related|returns|supply.chain|acoustic testing)\b", text):
        return (
            "What pre-production test results, supplier commitments, or yield data must "
            "be met before you commit to tooling or inventory spend?"
        )

    if re.search(r"\b(reduces?|saves?|improves?|faster|better|industry.leading|best.in.class|seamless|significant|unmatched|leading|strong|large|massive|revolutionary)\b", text):
        return (
            "What measurable customer outcome or independent benchmark supports this "
            "benefit, and what baseline are you comparing it with?"
        )

    if risk == "High":
        return (
            "What specific evidence would let an independent investor verify this claim, "
            "and what result would show that it is overstated?"
        )
    return (
        "What observable customer result or source evidence supports this statement, "
        "and what would falsify it?"
    )


def _targeted_question_candidates(sentence: str, risk: str) -> List[str]:
    """Return meaningfully different question angles for the claim's topic."""
    text = sentence.lower()
    base = _targeted_investor_question(sentence, risk)

    if re.search(
        r"\b(?:clinical|medical|therapeutic|knee pain|pain reduction|reduces? pain|"
        r"treats? symptoms?|diagnos(?:e|is)|injury reduction)\b",
        text,
    ):
        alternatives = [
            "What controlled trial measured the claimed outcome, and how did the results compare with a control or baseline group?",
            "Who assessed the result, which outcome was defined before testing, and can an independent reviewer reproduce it?",
            "What sample size and follow-up period support the claim, and were adverse or unchanged outcomes recorded too?",
            "How much of the observed change can be attributed to the product rather than selection effects or other interventions?",
            "Which independent evidence would be strong enough to justify using this outcome claim in customer-facing materials?",
        ]
    elif re.search(
        r"\b(?:comfort|fit consistency|fit-related|durability|wear trial|wear test|"
        r"footwear|sneakers?|shoes?|insole|outsole|shoe weight|prototype durability|"
        r"return rate)\b",
        text,
    ):
        alternatives = [
            "How many users and product units were tested, for how long, and what comparison group established the baseline?",
            "Which wear conditions, sizes, and user profiles were included, and what did the least successful group report?",
            "What measured return, defect, or fit-complaint data shows this design works after repeated real-world use?",
            "What pre-defined acceptance threshold did the product pass, and can the full test protocol and results be reviewed?",
            "Did an independent lab or representative customer cohort validate the claim, or is the evidence limited to an internal trial?",
            "How do the results compare with the closest competing product when both are tested under the same conditions?",
        ]
    elif re.search(
        r"\b(?:anc|active noise cancellation|noise control|microphone|mic array|"
        r"voice pickup|battery life|battery runtime|earbuds?|earable|"
        r"speaker diarisation|diarisation|transcription accuracy|meeting capture)\b",
        text,
    ):
        alternatives = [
            "How does performance change in wind, background noise, overlapping speech, or other conditions users actually encounter?",
            "Which benchmark and competing device were used, and can the reported result be reproduced on a fresh test set?",
            "What error rate, response delay, or battery drain did you measure during a complete real-world session?",
            "How many accents, environments, and device configurations were covered before you called this performance reliable?",
            "What failure cases remain, and how frequently do they occur outside the controlled demo environment?",
        ]
    elif re.search(r"\b(?:raising|raise|funding ask|use of funds|capital ask)\b", text):
        alternatives = [
            "Break this raise into funded milestones: what will be delivered at each stage, and how many months of runway does each tranche buy?",
            "Which outcomes must this capital produce before you return to investors, and what evidence will demonstrate they were achieved?",
            "How did you calculate the amount being raised, and which planned costs or milestones would you cut if less capital arrived?",
            "What is the first measurable proof point this round is meant to unlock, and what happens if it slips?",
            "Why is this raise the right size for the next stage rather than simply extending the current runway?",
            "If the round closes below target, which commitment changes first and which milestone remains achievable?",
        ]
    elif re.search(r"\b(revenue|forecast|financial plan|ebitda|runway|valuation|arr|mrr|year three|fy\d{2})\b", text):
        alternatives = [
            "Which inputs contribute most to the forecast, and what historical or signed-customer data supports each one?",
            "If conversion or pricing misses plan, how far does the revenue outcome fall and when would you know?",
            "Show how the projection changes under a conservative case; which assumption has the largest effect on gross profit?",
            "How much of this outlook comes from contracted demand versus pipeline or untested assumptions?",
            "What actual operating data anchors the forecast, and where does the model rely on extrapolation?",
            "Which metric has to improve first for the forecast to hold, and what evidence suggests that improvement is achievable?",
        ]
    elif re.search(r"\b(tam|sam|som|market size|market share|penetration|beachhead|trillion|billion|market estimate|market sizing)\b", text):
        alternatives = [
            "How did you size this opportunity from reachable buyers upward, and which customer segment is excluded from the calculation?",
            "What third-party source supports the market figure, and how did you reconcile differences across sources?",
            "At your proposed price, how many customers would you need to win to reach this share—and what acquisition capacity supports that?",
            "Which adoption or pricing assumption drives the market estimate most, and how sensitive is the result to changing it?",
            "What evidence says customers in this segment will switch from existing alternatives rather than merely express interest?",
            "Where does the initial beachhead sit within the wider market, and what data validates demand there today?",
        ]
    elif re.search(r"\b(customer-facing teams|ideal customer profile|target customer|buyer persona|teams with .{0,30}weekly calls|buyers who need)\b", text):
        alternatives = [
            "What recurring customer pain have you observed in discovery, and how many target buyers have agreed to pay to solve it?",
            "Which buyer interviews, purchase commitments or workflow records show this segment has urgent need rather than interest?",
            "How did you select this customer profile, and what evidence would make you narrow or change it?",
            "Who owns the budget for this problem, how is the purchase approved, and what has confirmed that process?",
            "What makes this segment more likely to buy now than adjacent customer groups, and what have real prospects demonstrated?",
        ]
    elif re.search(r"(?:[₹$€£]\s?[\d,]+.{0,25}(?:/|\bper\b).{0,12}(?:month|seat)|\bper month\b|\bsubscription(?: attach rate)?\b|\bfair-use\b|\bunlimited meeting minutes\b|\bpricing tier\b)", text):
        if re.search(r"\b(?:attach rate|retention|renewal|churn|cohort)\b", text):
            alternatives = [
                "How do activation, usage and renewal differ by subscriber cohort, and which pattern supports the assumed attach rate?",
                "What share of customers renew after the first billing period, and which usage behaviours explain the difference?",
                "Are retention and churn stable across plans, or is the headline rate being driven by a small group of heavy users?",
                "What cohort data would show that recurring demand persists after the initial trial or onboarding period?",
                "How sensitive is recurring revenue to a modest increase in churn, and what observed data supports the current assumption?",
            ]
        else:
            alternatives = [
                "What willingness-to-pay evidence sits behind this tier, and how does conversion change at nearby price points?",
                "Have real customers paid this amount yet, or is it still a pricing hypothesis? What does a live test show?",
                "After serving different usage levels, what gross margin remains for each plan and where is the break-even point?",
                "Which competitor or substitute sets the reference price, and what benefit convinces buyers to pay a premium?",
                "What would a small pricing experiment need to show before you commit to this rate across the customer base?",
                "At the heaviest expected usage, do support and infrastructure costs still leave a sustainable margin?",
            ]
    elif re.search(r"\b(unit economics|cogs|gross margin|contribution|manufacturing cost|asp|fulfilment|fulfillment|warranty|cost per|unit cost)\b", text):
        alternatives = [
            "Which fully loaded costs are included in contribution margin, from returns and fulfilment to support and channel fees?",
            "What supplier quotes or observed production costs substantiate these per-unit assumptions at the volume you project?",
            "How does the margin change with a lower realised price, higher returns or a more expensive acquisition channel?",
            "What is the payback period after gross margin and churn are accounted for, and which cohort proves it?",
            "Which cost components are measured today and which are estimates that still need validation?",
            "What happens to contribution per unit at twice the expected return rate or at a lower-than-planned selling price?",
        ]
    elif re.search(r"\b(?:blended cac|customer acquisition cost|cost per acquisition|search and referral loops)\b", text):
        alternatives = [
            "What do acquisition costs look like by channel and cohort, and which source has produced repeatable paying customers?",
            "Once referrals saturate, what paid acquisition cost do you expect—and is the payback still commercially viable?",
            "How much of the target depends on organic loops, and what measured referral data supports that assumption?",
            "What evidence connects this CAC target to observed conversion, retention and gross margin rather than clicks alone?",
            "Which channel has delivered the most durable customers so far, and how does its fully loaded acquisition cost compare?",
        ]
    elif re.search(r"\b(interviews?|waitlist|sign-ups?|paid units|paid deposits|pilot|beta|traction|activation|retention|conversion|weekly active|active users|paying customers|customer count|customers acquired|users acquired)\b", text):
        alternatives = [
            "Of the people who signed up or joined the waitlist, how many completed a meaningful action and how many paid?",
            "Can you show cohort-level activation and retention, not just cumulative registrations, and how have they changed over time?",
            "What share of pilot users converted to ongoing paid use, and what reasons explain those who did not?",
            "Are these results repeatable across customer segments, or driven by one-off pilots or founder-led relationships?",
            "Which independent records verify this traction—billing data, signed contracts, usage logs or shipped units?",
            "What funnel stage loses the most prospective customers, and what measured evidence points to the cause?",
        ]
    elif re.search(r"\b(founder|co-founder|founding team|leadership team|team lacks|team needs|hiring|hire|leader|capability gap|advisor|fractional operations|privacy counsel|former product lead|owns customer discovery|supply-chain leader|operations lead)\b", text):
        alternatives = [
            "Which gap is most likely to delay the next milestone, and who has clear ownership of closing it?",
            "What experience on the current team demonstrates the ability to execute this part of the plan, and what is still missing?",
            "Will this capability be hired, contracted or partnered for; what is the cost and deadline for securing it?",
            "If the role remains unfilled for another quarter, which delivery or commercial target is most exposed?",
            "Which responsibilities currently lack a clear owner, and how will you prevent that gap from affecting delivery?",
        ]
    elif re.search(r"\b(competitors?|competition|incumbents?|moat|defensib|patent|competitive advantage|clone|copy)\b", text):
        alternatives = [
            "Why would a customer choose this product over the strongest incumbent or a low-cost substitute today?",
            "What would stop a competitor with a larger distribution network from copying the key feature or bundling it?",
            "Which customer behaviour or asset gets harder for rivals to replicate as the business grows, and what proves it exists now?",
            "Have you tested switching against real alternatives, and what did buyers say would make them change?",
            "If an incumbent matched your core feature next quarter, what advantage would still give customers a reason to stay?",
        ]
    elif re.search(r"\b(transcription|diarisation|accuracy|latency|sub-\d|model quality|ai reliability|algorithm|benchmark|response time|technical performance)\b", text):
        alternatives = [
            "What test set, sample size and baseline were used, and can an independent party reproduce the reported result?",
            "How does performance hold up outside the best-case demo—under noisy inputs, peak load or unfamiliar users?",
            "Which metric defines acceptable performance in production, and what proportion of real-world cases currently meet it?",
            "What evidence shows this result generalises beyond the training or pilot environment?",
            "Where does the system fail most often today, and how are those failures represented in the reported average?",
        ]
    elif re.search(r"\b(app connectivity|meeting-platform integration|fully offline|universal integration|cloud ai|user consent|privacy|recording|security|deletion|encrypted|regulatory|compliance|legal)\b", text):
        alternatives = [
            "Walk through what the product does when consent is missing or revoked; how has that path been tested?",
            "Which data is collected, where is it retained, and what independently verified controls govern access and deletion?",
            "What is the failure mode if a platform integration or network connection is unavailable, and how is user data protected?",
            "Which regulatory or security requirement is most consequential for launch, and what audit or test confirms readiness?",
            "How do you demonstrate that permissions and deletion requests are respected end-to-end across supported platforms?",
        ]
    elif re.search(r"\b(supplier|manufactur|production|inventory|tooling|yield|hardware|battery|fit-related|returns|supply.chain|acoustic testing)\b", text):
        alternatives = [
            "What production trial has confirmed yield and quality at realistic volumes, rather than a prototype run?",
            "Which supplier commitments are documented, and what is the fallback if lead times or defect rates exceed plan?",
            "How would higher scrap, returns or component costs affect unit economics before you commit to tooling?",
            "What milestones must be met before inventory spend is released, and who signs off on each one?",
            "How will you confirm manufacturability, reliability and repeatable quality before increasing production volume?",
        ]
    elif re.search(r"\b(reduces?|saves?|improves?|faster|better|industry.leading|best.in.class|seamless|significant|unmatched|leading|strong|large|massive|revolutionary)\b", text):
        alternatives = [
            "What outcome can a customer measure before and after using the product, and how large is the observed difference?",
            "Which controlled comparison or independent benchmark supports the claimed improvement over the current workflow?",
            "What was the starting baseline, how many cases were measured, and did the improvement persist across users?",
            "What evidence would convince a sceptical buyer that this benefit is material enough to change purchasing behaviour?",
            "How does the benefit hold up for customers outside the pilot group, and what evidence rules out selection bias?",
        ]
    elif risk == "High":
        alternatives = [
            "Which outside evidence would most directly confirm this statement, and what would count as a failure?",
            "What hidden assumption must hold for this claim to be true, and how will you test it before scaling?",
            "How would you turn this statement into a measurable experiment with a clear pass/fail threshold?",
            "What would a sceptical investor ask to see in diligence before accepting this claim as credible?",
            "If this claim is overstated, which business decision is affected first, and what early signal would reveal that?",
        ]
    else:
        alternatives = [
            "What independent observation would support this statement, and what result would make you reconsider it?",
            "Which assumption is doing the most work in this claim, and how can it be tested with real customers?",
            "What evidence would distinguish a repeatable result from an optimistic interpretation of early data?",
            "How would you design a small, measurable test of this claim before relying on it in the business plan?",
            "What external record or customer outcome could verify the statement without relying on internal estimates?",
        ]

    # Preserve the existing topic-aware rule as one candidate, then choose from
    # alternatives whose evidence angle and sentence structure can differ.
    return list(dict.fromkeys([base, *alternatives]))


def _question_key(question: str) -> str:
    """Normalize punctuation and casing so cosmetic edits cannot evade deduplication."""
    return " ".join(re.findall(r"[a-z0-9]+", (question or "").lower()))


_QUESTION_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "if", "in", "into", "is", "it", "its", "of", "on", "or",
    "our", "rather", "should", "that", "the", "their", "them", "then", "this",
    "those", "to", "under", "was", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}


def _question_similarity(first: str, second: str) -> float:
    """Approximate wording similarity to prefer genuine rephrasings, not token swaps."""
    first_key = _question_key(first)
    second_key = _question_key(second)
    if first_key == second_key:
        return 1.0
    first_words = {
        word for word in first_key.split()
        if word not in _QUESTION_STOP_WORDS and len(word) > 2
    }
    second_words = {
        word for word in second_key.split()
        if word not in _QUESTION_STOP_WORDS and len(word) > 2
    }
    union = first_words | second_words
    overlap = len(first_words & second_words) / len(union) if union else 0.0
    # Word-set overlap measures shared subject matter; sequence overlap catches
    # templates with the same sentence shape but a few substituted nouns.
    from difflib import SequenceMatcher

    sequence = SequenceMatcher(None, first_key, second_key).ratio()
    return max(overlap, sequence)


def _fallback_question_candidates(sentence: str) -> List[str]:
    """Provide additional full rephrasings if a topic bank has already been used."""
    focus = re.sub(r"\s+", " ", sentence).strip().rstrip(".?!")
    if len(focus) > 110:
        focus = focus[:107].rsplit(" ", 1)[0] + "…"
    focus = focus or "this business claim"
    quoted_focus = f"“{focus}”"
    return [
        f"What independent evidence could confirm or disprove {quoted_focus}, and who could verify it?",
        f"Which assumption behind {quoted_focus} is most uncertain, and what is the quickest credible way to test it?",
        f"How would you turn {quoted_focus} into a real-world pilot with a measurable pass/fail threshold?",
        f"If {quoted_focus} falls short, which operating or commercial decision changes first?",
        f"What data would you put in an investor diligence room to substantiate {quoted_focus}, beyond internal projections?",
        f"Which downside case have you modelled for {quoted_focus}, and what early signal would trigger a course correction?",
        f"How does {quoted_focus} compare with an independently sourced baseline rather than an internal target?",
        f"What customer, supplier or usage record would most directly validate {quoted_focus} under scrutiny?",
        f"What result would make you revise {quoted_focus}, and how soon could you obtain that evidence?",
        f"Before scaling {quoted_focus}, what small experiment would prove the core assumption is sound?",
        f"Which part of {quoted_focus} depends most on an untested assumption, and what evidence would resolve it?",
        f"What would a sceptical buyer need to observe first to trust {quoted_focus}?",
    ]


def _select_diverse_question(
    candidates: List[str],
    previous_questions: List[str],
    sentence: str,
) -> str:
    """Choose a random option among the least-similar unused questions."""
    history_keys = {_question_key(question) for question in previous_questions}
    unique_candidates = list(dict.fromkeys(
        question.strip() for question in candidates if question and question.strip()
    ))
    available = [
        question for question in unique_candidates
        if _question_key(question) not in history_keys
    ]
    # If every unused topic template is already too close to an earlier
    # question, add claim-specific options so the selector can change the angle
    # and sentence structure instead of accepting a superficial rewording.
    if previous_questions and available:
        best_topic_distance = min(
            max(_question_similarity(question, previous) for previous in previous_questions)
            for question in available
        )
        if best_topic_distance > 0.52:
            fallback = _fallback_question_candidates(sentence)
            available_keys = {_question_key(question) for question in available}
            available.extend(
                question for question in fallback
                if _question_key(question) not in history_keys
                and _question_key(question) not in available_keys
            )

    if not available:
        fallback = _fallback_question_candidates(sentence)
        available = [
            question for question in fallback
            if _question_key(question) not in history_keys
        ]

    # For decks with many near-identical claims, topic banks plus contextual
    # fallbacks should suffice. Keep a final unique, semantically relevant option
    # rather than ever returning a duplicate question verbatim.
    if not available:
        focus = re.sub(r"\s+", " ", sentence).strip().rstrip(".?!")
        if len(focus) > 80:
            focus = focus[:77].rsplit(" ", 1)[0] + "…"
        focus = focus or "this claim"
        for framing in (
            f"What independently verifiable result would establish whether {focus} is accurate?",
            f"How could a third party test {focus} without relying on your internal estimates?",
            f"Which measurable outcome would make you confident enough to stand behind {focus}?",
        ):
            if _question_key(framing) not in history_keys:
                available = [framing]
                break
    if not available:
        # Extremely repetitive decks can exceed every prepared wording. This
        # clause is a last resort and changes the whole question structure.
        ordinal = len(previous_questions) + 1
        available = [
            f"During diligence, what evidence would independently test claim {ordinal}: {sentence.strip().rstrip('.?!')}?"
        ]

    if not previous_questions:
        return random.choice(available)

    scored = [
        (
            max(_question_similarity(question, previous) for previous in previous_questions),
            question,
        )
        for question in available
    ]
    best_score = min(score for score, _ in scored)
    # A small band allows randomness without choosing a near-copy over a
    # materially different alternative.
    best = [question for score, question in scored if score <= best_score + 0.06]
    return random.choice(best)


def generate_objection_question(
    sentence: str,
    risk: str,
    manager: ModelManager,
    previous_questions: Optional[List[str]] = None,
) -> Optional[str]:
    """Generate one targeted question for Moderate/High claims; diversify a whole deck."""
    if risk == "Low":
        return None

    history = previous_questions if previous_questions is not None else []
    candidates = _targeted_question_candidates(sentence, risk)

    # FLAN-T5 remains opt-in. Give it recent context, but validate and diversify
    # its response just like rule-based candidates; never trust it to deduplicate.
    if (
        getattr(manager, "use_flan_t5", False)
        and manager.generator_loaded
        and manager.generator_model is not None
        and manager.generator_tokenizer is not None
    ):
        try:
            recent_questions = history[-6:]
            previous_context = "\n".join(f"- {question}" for question in recent_questions)
            prompt = (
                "You are a skeptical seed-stage venture capitalist. Write exactly one "
                "concise, claim-specific question seeking evidence about the weakest "
                "assumption, metric, customer validation, economics, competition, or "
                "implementation risk. Choose a genuinely different line of inquiry and "
                "sentence structure from every previous question; do not repeat or lightly "
                "paraphrase them. Do not ask for a title or a generic VC objection.\n"
                f"Risk: {risk}\nClaim: {sentence}\n"
                f"Questions already used in this deck:\n{previous_context or '(none)'}\n"
                "Return one question only:\n"
            )
            inputs = manager.generator_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(manager.device)
            with torch.no_grad():
                outputs = manager.generator_model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.92,
                    no_repeat_ngram_size=3,
                )
            generated = manager.generator_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            if not generated.endswith("?"):
                generated += "?"
            if _is_useful_generated_question(generated):
                candidates.insert(0, generated)
            else:
                logger.warning("Discarding generic/irrelevant FLAN-T5 question: %r", generated)
        except Exception as exc:
            logger.warning("FLAN-T5 question generation failed; using targeted fallback: %s", exc)

    return _select_diverse_question(candidates, history, sentence)

def analyze_sentences(
    sentences: List[str],
    raw_text: Optional[str] = None,
    validate_document: bool = False,
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

    # Document validation is enforced by the upload/API layer.
    # Keep it optional here so unit tests and internal callers can test
    # claim screening and risk classification independently.
    if validate_document:
        document_warning = document_warning_for_text(raw_text)
        if document_warning:
            pipeline_info["note"] = (
                "Risk scoring was skipped because the document does not appear "
                "to be a startup pitch deck."
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
    previous_questions: List[str] = []
    for idx, (claim, risk, score) in enumerate(candidates, start=1):
        if risk == "Low":
            summary_counts["low_risk"] += 1
        elif risk == "Moderate":
            summary_counts["moderate_risk"] += 1
        elif risk == "High":
            summary_counts["high_risk"] += 1

        question = generate_objection_question(
            claim, risk, manager, previous_questions=previous_questions
        )
        if question is not None:
            previous_questions.append(question)
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
