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

    if re.search(r"\b(tam|sam|som|market size|market share|penetration|beachhead|trillion|billion|market estimate|market sizing)\b", text):
        return (
            "What bottom-up assumptions for the reachable customer segment, price, "
            "and adoption rate support this market estimate?"
        )

    if re.search(r"\b(interviews?|waitlist|sign-ups?|paid units|paid deposits|pilot|beta|traction|activation|retention|conversion|weekly active|customers?|users?)\b", text):
        return (
            "How many of these interested users or pilot participants converted to "
            "paying customers, and what evidence supports that conversion rate?"
        )

    if re.search(r"\b(revenue|forecast|financial plan|ebitda|runway|crore|million|billion|funding|raise|raising|valuation|year three|fy\d{2})\b", text):
        return (
            "Which assumptions for sales volume, realised price, conversion, and gross "
            "margin drive this forecast, and what evidence validates them?"
        )

    if re.search(r"\b(unit economics|cogs|gross margin|contribution|manufacturing cost|asp|fulfilment|fulfillment|warranty|cost per|unit cost|pricing|price)\b", text):
        return (
            "Do these unit economics include discounts, returns, fulfilment, warranty, "
            "channel fees, and acquisition cost, and which supplier quotes support them?"
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

    if re.search(r"\b(privacy|consent|recording|security|retention|deletion|encrypted|regulatory|compliance|legal)\b", text):
        return (
            "What concrete controls or independent review demonstrate that this privacy "
            "and compliance approach works in the intended deployment?"
        )

    if re.search(r"\b(supplier|manufactur|production|inventory|tooling|yield|hardware|battery|fit-related|returns|supply.chain|acoustic testing)\b", text):
        return (
            "What pre-production test results, supplier commitments, or yield data must "
            "be met before you commit to tooling or inventory spend?"
        )

    if re.search(r"\b(team|founder|hire|hiring|leader|capability gap|advisor|operations|supply-chain)\b", text):
        return (
            "Which critical capability is currently missing, who will own it, and what "
            "is the hiring or partner timeline before the next milestone?"
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
