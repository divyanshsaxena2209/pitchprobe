"""Layout-aware sentence and claim-candidate segmentation for pitch-deck PDFs."""

import re
from typing import List

import spacy

try:
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = spacy.blank("en")
    _nlp.add_pipe("sentencizer")


_METRIC_PATTERN = re.compile(
    r"\b(?:ARR|MRR|TAM|SAM|SOM|CAC|LTV|GMV|ROI|revenue|customers?|"
    r"users?|growth|retention|churn|margin|sales|waitlist|deposits?)\b",
    re.IGNORECASE,
)
_VALUE_PATTERN = re.compile(r"[$€£₹%]|\b\d+(?:[,.]\d+)?\b")


def _normalise_pdf_text(text: str) -> str:
    """Repair common PDF glyph and spacing artifacts before segmenting."""
    value = re.sub(r"\(cid:\d+\)", " • ", text or "", flags=re.IGNORECASE)
    value = re.sub(r"(?<!\w)fi(?!\w)", "→", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<![A-Za-z])n(?=\s?\d)", "₹", value)
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    value = re.sub(r"[-_=]{3,}", " ", value)
    return value


def split_into_sentences(raw_text: str) -> List[str]:
    """Split within layout-preserved text blocks, never across neighbouring columns."""
    if not raw_text or not raw_text.strip():
        return []

    normalized = _normalise_pdf_text(raw_text)
    # The extractor separates spatially independent paragraphs/columns with blank
    # lines. Keep those boundaries; splitting the whole PDF at once can join cells.
    blocks = re.split(r"\n\s*\n+", normalized)
    sentences: List[str] = []

    for block in blocks:
        block = re.sub(r"[\r\t]+", " ", block).strip()
        if not block:
            continue

        # Bullet glyphs from embedded slide fonts are delimiters, not words.
        # Separating them stops a long row of unrelated bullets becoming one claim.
        bullet_parts = re.split(r"\s*•\s*", block)
        for part in bullet_parts:
            part = re.sub(r"\s+", " ", part).strip(" \t\r\n-–—")
            if not part:
                continue

            doc = _nlp(part)
            for sent in doc.sents:
                candidate = sent.text.strip()
                candidate = re.sub(r"^[•\-\*\d\.\)\s]+", "", candidate).strip()
                candidate = re.sub(r"\s+", " ", candidate)
                if not candidate:
                    continue

                words = candidate.split()
                has_metric = bool(
                    _METRIC_PATTERN.search(candidate)
                    and _VALUE_PATTERN.search(candidate)
                )
                if (len(candidate) >= 12 and len(words) >= 3) or (
                    has_metric and len(candidate) >= 6
                ):
                    sentences.append(candidate)

    # Preserve order while removing exact duplicates introduced by repeated slide
    # footers or text extraction overlaps.
    deduplicated = []
    seen = set()
    for sentence in sentences:
        key = re.sub(r"\s+", " ", sentence).casefold()
        if key not in seen:
            seen.add(key)
            deduplicated.append(sentence)
    return deduplicated
