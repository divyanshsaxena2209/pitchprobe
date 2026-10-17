import logging
import os
import re
import threading
from typing import Dict, List, Optional, Tuple
import torch

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
                        tok = AutoTokenizer.from_pretrained(self.classifier_name, local_files_only=True)
                        model = AutoModelForSequenceClassification.from_pretrained(
                            self.classifier_name, local_files_only=True
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
                        tok = AutoTokenizer.from_pretrained(self.generator_name, local_files_only=True)
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            self.generator_name, local_files_only=True
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
    """Classify sentence risk using cross-encoder NLI hypotheses or calibrated heuristics."""
    if manager.classifier_loaded and manager.classifier_model and manager.classifier_tokenizer:
        try:
            # Pair sentence premise with each risk hypothesis
            pairs = [
                (sentence, NLI_HYPOTHESES["Low"]),
                (sentence, NLI_HYPOTHESES["Moderate"]),
                (sentence, NLI_HYPOTHESES["High"]),
            ]
            features = manager.classifier_tokenizer(
                pairs, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(manager.device)

            with torch.no_grad():
                outputs = manager.classifier_model(**features)
                if outputs.logits.shape[-1] == 1:
                    scores = outputs.logits.squeeze(-1)
                else:
                    scores = outputs.logits[:, -1]
                probs = torch.softmax(scores, dim=0).cpu().tolist()

            tier_map = ["Low", "Moderate", "High"]
            best_idx = int(torch.argmax(torch.tensor(probs)).item())
            return tier_map[best_idx], round(float(probs[best_idx]), 4)
        except Exception as e:
            logger.warning("NLI inference failed for sentence, falling back to heuristics: %s", e)

    # Calibrated startup heuristic fallback (DA1 Section 5 Fallback Plan)
    s_lower = sentence.lower()

    high_hits = sum(1 for p in HYPERBOLIC_PATTERNS if re.search(p, s_lower))
    grounded_hits = sum(1 for p in GROUNDED_PATTERNS if re.search(p, s_lower))
    vague_hits = sum(1 for p in VAGUE_PATTERNS if re.search(p, s_lower))

    if high_hits > 0:
        confidence = min(0.78 + (high_hits * 0.08), 0.96)
        return "High", round(confidence, 2)
    elif grounded_hits > 0 and vague_hits == 0:
        confidence = min(0.80 + (grounded_hits * 0.06), 0.98)
        return "Low", round(confidence, 2)
    elif vague_hits > 0:
        confidence = min(0.74 + (vague_hits * 0.07), 0.92)
        return "Moderate", round(confidence, 2)
    else:
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


def analyze_sentences(sentences: List[str]) -> Tuple[List[Dict], Dict[str, int], Dict[str, str]]:
    """Execute two-stage pipeline: Risk classification followed by gated investor question generation."""
    manager = ModelManager.get_instance()

    classified_results = []
    summary_counts = {"low_risk": 0, "moderate_risk": 0, "high_risk": 0}

    for idx, sentence in enumerate(sentences, start=1):
        # Stage 1: Risk Classification
        risk, score = classify_sentence_nli(sentence, manager)

        # Update summary counts
        if risk == "Low":
            summary_counts["low_risk"] += 1
        elif risk == "Moderate":
            summary_counts["moderate_risk"] += 1
        elif risk == "High":
            summary_counts["high_risk"] += 1

        # Stage 2: Gated Question Generation (None for Low risk)
        question = generate_objection_question(sentence, risk, manager)

        classified_results.append({
            "id": idx,
            "text": sentence,
            "risk": risk,
            "score": score,
            "question": question,
        })

    mode = (
        "transformer_local"
        if (manager.classifier_loaded and manager.generator_loaded)
        else "calibrated_domain_heuristics"
    )

    pipeline_info = {
        "classifier": manager.classifier_name,
        "generator": manager.generator_name,
        "mode": mode,
        "note": "Sentence-level risk stratification gates investor question generation per DA1 architecture.",
    }

    return classified_results, summary_counts, pipeline_info
