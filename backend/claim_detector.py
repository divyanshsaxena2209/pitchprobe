"""High-precision investor-claim screening before risk classification.

The screener deliberately prefers precision over scoring every sentence. It
uses deterministic exclusions for obvious document noise, strong business-claim
signals for clear cases, and the existing NLI model for ambiguous cases.
"""

import re
from typing import Optional

import torch


DOCUMENT_WARNING = (
    "This PDF appears to be an academic assignment or research report rather "
    "than a startup pitch deck. No investor-facing claims were scored. Upload "
    "a startup pitch deck containing product, market, traction, customer, or "
    "financial claims for a useful investor review."
)

ADMIN_DOCUMENT_MARKERS = [
    r"\bcourse\s*code\b",
    r"\bsubmission\s+deadline\b",
    r"\bstudent\s+name\b",
    r"\bregistration\s*(?:no\.?|number)\b",
    r"\bcontribution\s+matrix\b",
    r"\btotal\s+marks?\b",
    r"\bassessment\s+rubric\b",
    r"\bassignment\s+rubric\b",
    r"\bacademic\s+year\s*/?\s*semester\b",
    r"\bB\.?\s*Tech\b",
    r"\bfaculty\b",
]

RESEARCH_DOCUMENT_MARKERS = [
    r"\bliterature survey\b",
    r"\bresearch gaps?\b",
    r"\bproposed system architecture\b",
    r"\bevaluation metrics?\b",
    r"\bbaseline comparison\b",
    r"\breference formatting\b",
    r"\bcontribution matrix\b",
    r"\bmacro[- ]?f1\b",
    r"\bbertscore\b",
    r"\brouge[- ]?l\b",
    r"\b(?:IEEE|ACL|EMNLP|arXiv)\b",
    r"\bfree[- ]tier colab\b",
    r"\bVRAM\b",
    r"\bpapers?\s*\[\s*\d+",
    r"\btable\s+\d+\s+(?:summarises|summarizes|shows|compares)\b",
]

BUSINESS_TERMS = re.compile(
    r"\b(?:revenue|sales|ARR|MRR|CAC|LTV|payback|unit economics|"
    r"customers?|users?|subscribers?|retention|churn|conversion|"
    r"growth|traction|market|TAM|SAM|SOM|market share|pricing|"
    r"price|profit|margin|runway|funding|raised|valuation|"
    r"competitors?|competition|competitive advantage|moat|patent|"
    r"customers? acquired|contracts?|paid pilots?|waitlist|"
    r"product|platform|solution|service|business|founders?|"
    r"small businesses|clinics?|hospitals?|merchants?|teams?|"
    r"ROI|GMV|orders?|bookings?|active accounts?)\b",
    re.IGNORECASE,
)

ASSERTION_TERMS = re.compile(
    r"\b(?:is|are|has|have|had|will|can|could|achieved|reached|"
    r"generated|grew|grown|increased|increases|decreased|decreases|"
    r"reduced|reduces|cut|saves?|spends?|costs?|launched|signed|"
    r"raised|serves?|retains?|offers?|provides?|enables?|helps?|"
    r"solves?|captures?|automates?|improves?|delivers?|"
    r"expects?|forecast|projects?|targets?|plans? to|"
    r"converts?|processes?|supports?|eliminates?)\b",
    re.IGNORECASE,
)

PRODUCT_SUBJECT_ACTION = re.compile(
    r"\b(?:we|our|the product|the platform|the solution|our technology|"
    r"our system|our service|the company)\b.{0,100}"
    r"\b(?:helps?|enables?|provides?|offers?|reduces?|cuts?|saves?|"
    r"automates?|improves?|delivers?|serves?|increases?|"
    r"eliminates?|grew|reached|achieved|will|has|have|is|are)\b",
    re.IGNORECASE,
)

NON_CLAIM_PATTERNS = [
    # Section headings and document furniture.
    r"^\s*(?:references?|bibliography|appendix|table of contents)\s*:?\s*$",
    r"^\s*(?:table|figure)\s+\d+\b",
    r"^\s*\[\s*\d{1,3}\s*\]\s*[A-Z]",
    r"^\s*(?:literature survey|research gap|problem identification|"
    r"proposed system architecture|evaluation metrics|feasibility note|"
    r"contribution matrix|baseline comparison|reference formatting|"
    r"novelty placement)\b",
    # Academic or administrative content that is not an investor claim.
    r"\b(?:course\s*code|submission\s+deadline|student\s+name|"
    r"registration\s*(?:no\.?|number)|academic\s+year\s*/?\s*semester|"
    r"assessment\s+rubric|assignment\s+rubric|contribution\s+matrix|"
    r"total\s+marks?)\b",
    r"^\s*(?:this paper|this study|existing work|papers?\s*\[\s*\d+)",
    r"\b(?:the primary stakeholder is|the decision supported is|"
    r"table \d+ summarises|table \d+ summarizes|"
    r"selected from IEEE|reference formatting)\b",
    # Reference entries and technical research reporting, rather than
    # investor-facing assertions.
    r"\b(?:macro[- ]?f1|bertscore|rouge[- ]?l|free[- ]tier colab|"
    r"reference benchmark|annotation set|peer-reviewed preprint)\b",
]


def document_warning_for_text(raw_text: Optional[str]) -> Optional[str]:
    """Detect likely coursework/research documents before analyzing sentences."""
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.lower()
    admin_hits = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in ADMIN_DOCUMENT_MARKERS)
    research_hits = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in RESEARCH_DOCUMENT_MARKERS)

    # Two or more distinct administrative markers strongly suggest coursework.
    # Academic markers require a higher threshold to avoid rejecting a technical
    # startup deck that happens to mention a benchmark or architecture.
    if admin_hits >= 2 or research_hits >= 6:
        return DOCUMENT_WARNING
    return None


def is_obvious_non_claim(text: str) -> bool:
    """Remove headings, references, administrative details, and research noise."""
    candidate = re.sub(r"\s+", " ", text or "").strip()
    if len(candidate) < 12 or len(candidate.split()) < 3:
        return True

    for pattern in NON_CLAIM_PATTERNS:
        if re.search(pattern, candidate, re.IGNORECASE):
            return True

    # Short all-caps lines are usually headings. Keep quantified/assertive
    # headline claims such as "REVENUE GREW 35%" for analysis.
    letters = [char for char in candidate if char.isalpha()]
    is_all_caps = bool(letters) and all(char.isupper() for char in letters)
    has_claim_detail = bool(
        re.search(r"[$€£₹%]|\b\d+(?:\.\d+)?\b", candidate)
        or ASSERTION_TERMS.search(candidate)
    )
    if is_all_caps and len(candidate.split()) <= 14 and not has_claim_detail:
        return True

    return False


def _entailment_probability(text: str, hypothesis: str, manager: object) -> float:
    """Return the NLI entailment probability for one sentence/hypothesis pair."""
    tokenizer = getattr(manager, "classifier_tokenizer", None)
    model = getattr(manager, "classifier_model", None)
    if tokenizer is None or model is None:
        raise RuntimeError("NLI classifier is not loaded")

    features = tokenizer(
        [(text, hypothesis)],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(getattr(manager, "device", "cpu"))

    with torch.no_grad():
        logits = model(**features).logits

    id2label = {
        int(key): str(value).lower()
        for key, value in model.config.id2label.items()
    }
    entailment_ids = [
        index for index, label in id2label.items()
        if "entail" in label
    ]
    if len(entailment_ids) != 1:
        raise ValueError(f"Cannot identify NLI entailment label in {id2label}")

    return float(torch.softmax(logits, dim=-1)[0, entailment_ids[0]].item())


def is_investor_claim(text: str, manager: Optional[object] = None) -> bool:
    """Return True only for likely investor-relevant business/product claims."""
    candidate = re.sub(r"\s+", " ", text or "").strip()
    if is_obvious_non_claim(candidate):
        return False

    has_business_term = bool(BUSINESS_TERMS.search(candidate))
    has_assertion = bool(ASSERTION_TERMS.search(candidate))
    has_product_assertion = bool(PRODUCT_SUBJECT_ACTION.search(candidate))

    # High-confidence business claims do not need another model call.
    if (has_business_term and has_assertion) or has_product_assertion:
        return True

    # Use the already-loaded NLI model for ambiguous sentences; don't download
    # or initialize a second model solely for claim detection.
    if (
        manager is not None
        and getattr(manager, "classifier_loaded", False)
        and getattr(manager, "classifier_model", None) is not None
        and getattr(manager, "classifier_tokenizer", None) is not None
    ):
        claim_hypothesis = (
            "This sentence makes a substantive investor-facing claim about a "
            "startup's product, customers, market, traction, financials, "
            "competitive advantage, or future business performance."
        )
        non_claim_hypothesis = (
            "This sentence is explanatory background, administrative text, a "
            "heading, reference, citation, literature-review statement, or "
            "technical methodology detail rather than an investor-facing "
            "business or product claim."
        )
        try:
            claim_score = _entailment_probability(candidate, claim_hypothesis, manager)
            non_claim_score = _entailment_probability(candidate, non_claim_hypothesis, manager)
            return claim_score >= 0.45 and claim_score >= non_claim_score + 0.08
        except Exception:
            # The ordinary risk classifier has its own fallback. Claim screening
            # remains conservative if the model cannot score this sentence.
            return False

    # No model available: only admit text with recognizable business language
    # and an assertion. Unmatched sentences are not automatically "Moderate".
    return has_business_term and has_assertion
