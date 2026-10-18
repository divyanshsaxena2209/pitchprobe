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

    # Pre-clean excessive whitespace and common slide bullet noise.
    normalized_text = re.sub(r"[\r\t]+", " ", raw_text)
    # Some PDFs omit spaces between text runs (for example, "OurPlatformIs").
    # Restore recoverable CamelCase/acronym boundaries before sentence splitting.
    normalized_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", normalized_text)
    normalized_text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", normalized_text)
    # Replace slide boundary artifacts or long underscores/dashes
    normalized_text = re.sub(r"[-_=]{3,}", " ", normalized_text)

    doc = _nlp(normalized_text)
    sentences: List[str] = []

    for sent in doc.sents:
        s = sent.text.strip()
        # Clean bullet points, numbering, or quotes from sentence starts
        s = re.sub(r"^[•\-\*\d\.\)\s]+", "", s).strip()
        s = re.sub(r"\s+", " ", s)

        # Keep normal sentences plus compact but meaningful metric bullets
        # such as "TAM: $5B" or "MRR $20k".
        words = s.split()
        has_business_metric = bool(
            re.search(
                r"\b(?:ARR|MRR|TAM|SAM|SOM|CAC|LTV|GMV|ROI|revenue|"
                r"customers?|users?|growth|retention|churn|margin|sales)\b",
                s,
                re.IGNORECASE,
            )
            and re.search(r"[$€£₹%]|\b\d+(?:\.\d+)?\b", s)
        )
        if (len(s) >= 12 and len(words) >= 3) or (
            has_business_metric and len(s) >= 6
        ):
            sentences.append(s)

    return sentences
