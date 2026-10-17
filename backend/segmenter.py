import re
from typing import List
import spacy

# Load spaCy pipeline once globally
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    # Lightweight rule-based sentencizer fallback without external weights download
    _nlp = spacy.blank("en")
    _nlp.add_pipe("sentencizer")


def split_into_sentences(raw_text: str) -> List[str]:
    """Segment extracted text into clean, individual pitch claim sentences."""
    if not raw_text or not raw_text.strip():
        return []

    # Pre-clean excessive whitespace and common slide bullet noise
    normalized_text = re.sub(r"[\r\t]+", " ", raw_text)
    # Replace slide boundary artifacts or long underscores/dashes
    normalized_text = re.sub(r"[-_=]{3,}", " ", normalized_text)

    doc = _nlp(normalized_text)
    sentences: List[str] = []

    for sent in doc.sents:
        s = sent.text.strip()
        # Clean bullet points, numbering, or quotes from sentence starts
        s = re.sub(r"^[•\-\*\d\.\)\s]+", "", s).strip()
        s = re.sub(r"\s+", " ", s)

        # Exclude tiny fragments, pure numbers, or solitary header tokens (< 12 chars or < 3 words)
        words = s.split()
        if len(s) >= 12 and len(words) >= 3:
            sentences.append(s)

    return sentences
