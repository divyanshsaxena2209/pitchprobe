import logging
import os
import re
import threading
from typing import Dict, List, Optional, Tuple
import torch

from backend.claim_detector import document_warning_for_text, is_investor_claim

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier_tokenizer = None
        self.classifier_model = None
        self.generator_tokenizer = None
        self.generator_model = None

        self.classifier_name = "cross-encoder/nli-deberta-v3-base"
        self.generator_name = "google/flan-t5-small"

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
                    try:
                        logger.info("Checking local cache for %s...", self.classifier_name)
                        tok = AutoTokenizer.from_pretrained(self.classifier_name, local_files_only=False)
                        model = AutoModelForSequenceClassification.from_pretrained(
                            self.classifier_name, local_files_only=False
                        ).to(self.device)
                        model.eval()
                        self.classifier_tokenizer = tok
                        self.classifier_model = model
                        self.classifier_loaded = True
                        logger.info("Loaded classifier %s from local cache.", self.classifier_name)
                    except Exception:
                        logger.info("Local weights for %s not found; using calibrated heuristics.", self.classifier_name)

                if not self.generator_loaded:
                    try:
                        logger.info("Checking local cache for %s...", self.generator_name)
                        tok = AutoTokenizer.from_pretrained(self.generator_name, local_files_only=False)
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            self.generator_name, local_files_only=False
                        ).to(self.device)
                        model.eval()
                        self.generator_tokenizer = tok
                        self.generator_model = model
                        self.generator_loaded = True
                        logger.info("Loaded generator %s from local cache.", self.generator_name)
                    except Exception:
                        logger.info("Local weights for %s not found; using calibrated VC counter-questions.", self.generator_name)
            except Exception as e:
                logger.warning("Model loader encountered error: %s", e)

        if background:
            thread = threading.Thread(target=_loader, daemon=True)
            thread.start()
        else:
            _loader()



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

            # Each sentence/hypothesis pair gets one entailment score.
            entailment_scores = torch.softmax(logits, dim=-1)[
                :, entailment_ids[0]
            ]

            # Compare the three hypotheses and normalise for display.
            probabilities = torch.softmax(
                entailment_scores, dim=0
            ).cpu().tolist()

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



def generate_objection_question(sentence: str, risk: str, manager: ModelManager) -> Optional[str]:
    """Generate exactly one investor counter-question for Moderate/High sentences; Low returns None."""
    # Strict Gating Rule (DA1 Section 4.5 & Requirement 5):
    # Low-risk sentences NEVER receive a question.
    if risk == "Low":
        return None

    if manager.generator_loaded and manager.generator_model and manager.generator_tokenizer:
        try:
            prompt = f"Investor review: {sentence}. Generate one tough VC objection question:"
            inputs = manager.generator_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(manager.device)

            with torch.no_grad():
                outputs = manager.generator_model.generate(
                    **inputs,
                    max_new_tokens=48,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
                generated = manager.generator_tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                ).strip()

            if generated and len(generated) > 10:
                if not generated.endswith("?"):
                    generated += "?"
                return generated
        except Exception as e:
            logger.warning("FLAN-T5 question generation failed, using VC fallback template: %s", e)

    # Targeted investor objection generation heuristics conditioned on claim context
    s_lower = sentence.lower()

    if any(k in s_lower for k in ["market", "tam", "sam", "som", "trillion", "billion", "share"]):
        return "What bottom-up calculation or initial customer segment validates this market size projection?"
    elif any(k in s_lower for k in ["competitor", "competition", "monopoly", "only", "dominant"]):
        return "What structural moat or defensible IP prevents well-funded incumbents from cloning your solution?"
    elif any(k in s_lower for k in ["revenue", "growth", "scale", "sales", "arr", "mrr", "profit"]):
        return "What are your specific unit economics (CAC, LTV, and payback period) underpinning this trajectory?"
    elif any(k in s_lower for k in ["ai", "platform", "technology", "algorithm", "deep tech"]):
        return "How much of this technology relies on third-party APIs versus proprietary defensible architecture?"
    elif risk == "High":
        return "What audited operational milestones or customer pilot data substantiate this ambitious claim?"
    else:
        return "What concrete empirical evidence or customer metrics support this statement?"


def analyze_sentences(
    sentences: List[str],
    raw_text: Optional[str] = None,
) -> Tuple[List[Dict], Dict[str, int], Dict[str, str]]:
    """Screen for investor claims, classify only claims, then gate questions."""
    manager = ModelManager.get_instance()
    summary_counts = {"low_risk": 0, "moderate_risk": 0, "high_risk": 0}

    mode = (
        "transformer_local"
        if (manager.classifier_loaded and manager.generator_loaded)
        else "calibrated_domain_heuristics"
    )
    pipeline_info = {
        "classifier": manager.classifier_name,
        "generator": manager.generator_name,
        "mode": mode,
        "note": (
            "Claim-first screening filters non-claims before risk scoring; "
            "investor questions are generated only for Moderate/High claims."
        ),
        "document_warning": None,
    }

    # Detect coursework/research documents at document level. Scoring individual
    # sentences from those files would create misleading investor objections.
    document_warning = document_warning_for_text(raw_text)
    if document_warning:
        pipeline_info["note"] = (
            "Risk scoring was skipped because the document does not appear to "
            "be a startup pitch deck."
        )
        pipeline_info["document_warning"] = document_warning
        return [], summary_counts, pipeline_info

    # Stage 0: filter non-claims. The risk classifier must never assign a risk
    # tier simply because a sentence exists in the PDF.
    claims = [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in sentences
        if is_investor_claim(sentence, manager)
    ]

    if not claims:
        pipeline_info["note"] = (
            "No investor-facing claims were detected, so risk scoring was skipped."
        )
        pipeline_info["document_warning"] = (
            "No investor-facing claims were detected. Upload a startup pitch deck "
            "with product, market, traction, customer, or financial assertions."
        )
        return [], summary_counts, pipeline_info

    classified_results = []

    for idx, claim in enumerate(claims, start=1):
        # Stage 1: score detected claims only.
        risk, score = classify_sentence_nli(claim, manager)

        if risk == "Low":
            summary_counts["low_risk"] += 1
        elif risk == "Moderate":
            summary_counts["moderate_risk"] += 1
        elif risk == "High":
            summary_counts["high_risk"] += 1

        # Stage 2: low-risk claims get no question; Moderate/High get one.
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
