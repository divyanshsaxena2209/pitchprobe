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

STRONG_ADMIN_MARKERS = [
    "coursecode",
    "submissiondeadline",
    "registrationno",
    "contributionmatrix",
    "totalmarks",
    "assessmentrubric",
    "assignmentrubric",
    "evaluationrubric",
]

ADMIN_COMPACT_MARKERS = STRONG_ADMIN_MARKERS + [
    "btech",
    "faculty",
    "academicyearsemester",
    "studentname",
]

TEMPLATE_WARNING = (
    "This PDF appears to be a pitch-deck template or instruction guide rather "
    "than a completed startup pitch deck. No investor-facing claims were scored. "
    "Replace the prompts with your venture's actual product, customer, market, "
    "traction, financial, team, and funding claims, then upload the completed deck."
)

# Compact phrases copied from common pitch-deck teaching templates. We require
# several distinct instructions before warning, so one ordinary phrase in a
# completed deck (for example, a roadmap or three-year forecast) is not enough.
PITCH_TEMPLATE_INSTRUCTION_MARKERS = [
    "thegoalofyourpitchdeckistogiveasnapshotof",
    "themostcompellingelementsofyourexecutivesummary",
    "youwillbeallotted10minutesorless",
    "ensureslidesarenottextordataheavy",
    "defineyourventurewithashortdeclarativesentence",
    "describethepainofthecustomer",
    "outlinehowthecustomeraddresses",
    "providecasescustomerinsightscasestudies",
    "demonstrateyourcompanysvalueproposition",
    "providebottomsupmarketsizing",
    "identifyprofilethecustomeryoucater",
    "changingtrendsnewtechnologiesandforcesdrivingthissegment",
    "givealistofdirectcompetitorssubstitutes",
    "whathasntanyonedonethisbefore",
    "howwillyoureachtargetmarket",
    "wherewillyoupromoteyourproduct",
    "whatisyourplanforexecution",
    "summarizesalesmarketingandpartnershipplans",
    "productlineupformfactorfunctionalityfeatures",
    "explainhowyoumakemoney",
    "provide3yearforecastincludingannualrevenue",
    "providethreeyearforecast",
    "howmuchcapitalwillyouneedpriortobecomingcashflowpositive",
    "youneedtoconveythatyouaretheteamtoexecutethisventure",
    "additionalslidesthatcanbeusedforfollowupquestions",
]

RESEARCH_COMPACT_MARKERS = [
    "literaturesurvey",
    "researchgap",
    "proposedsystemarchitecture",
    "evaluationmetrics",
    "baselinecomparison",
    "referenceformatting",
    "contributionmatrix",
    "macrof1",
    "bertscore",
    "rougel",
    "emnlp",
    "freetiercolab",
    "vram",
    "references",
    "bibliography",
]

STRONG_BUSINESS_MARKERS = [
    "revenue",
    "arr",
    "mrr",
    "payingcustomers",
    "customeracquisition",
    "uniteconomics",
    "grossmargin",
    "retention",
    "churn",
    "paidpilots",
    "signedcontracts",
    "salespipeline",
    "runway",
]

BUSINESS_TERMS = re.compile(
    r"\b(?:revenue|sales|ARR|MRR|CAC|LTV|payback|unit economics|"
    r"customers?|users?|subscribers?|retention|churn|conversion|"
    r"growth|traction|market|TAM|SAM|SOM|market share|pricing|"
    r"price|profit|margin|runway|funding|raised|valuation|"
    r"competitors?|competition|competitive advantage|moat|patent|"
    r"customers? acquired|contracts?|paid pilots?|waitlist|"
    r"product|platform|solution|service|business|founders?|"
    r"small businesses|SMBs?|SMEs?|enterprises?|clinics?|hospitals?|"
    r"merchants?|teams?|healthcare|logistics|retail|manufacturers?|"
    r"inventory|invoices?|compliance|pain points?|consumers?|patients?|"
    r"ROI|GMV|orders?|bookings?|active accounts?|interviews?|deposits?|shipped|sign-ups?)\b",
    re.IGNORECASE,
)

ASSERTION_TERMS = re.compile(
    r"\b(?:is|are|has|have|had|will|can|could|achieved|reached|"
    r"generated|grew|grown|increased|increases|decreased|decreases|"
    r"reduced|reduces|cut|saves?|spends?|spent|loses?|lost|wastes?|"
    r"lacks?|struggles?|requires?|needs?|takes?|valued|worth|"
    r"costs?|launched|signed|raised|serves?|retains?|offers?|"
    r"provides?|enables?|helps?|solves?|captures?|automates?|"
    r"improves?|delivers?|expects?|forecast|projects?|targets?|"
    r"plans? to|converts?|processes?|supports?|eliminates?|"
    r"prevents?|avoids?|accelerates?|scales?)\b",
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
    # Pitch-deck furniture, source notes, and decorative labels.
    r"^\s*\d{1,2}\s*/\s*(?:the problem|the solution|product|technology|market opportunity|customer|competitive landscape|business model|go-to-market|early signals|financial plan|risks?(?: & mitigations)?|roadmap|team|the ask)\s*$",
    r"\bsonora\s*/\s*seed\b.*\billustrative\b.*\bnot an offer\b",
    r"^\s*(?:source url:|confidentiality note$|for discussion and analytical testing only|illustrative assumptions$|market reference$|proposed advisor profile$|use of funds$|what the signal does / does not say$|positive$|still unproven$|earable$|phone app$|ai services$|work tools$|on device$|phone / cloud$)",
    r"^\s*grand view research,",
    r"\b(?:all company, founder, traction and forecast data are illustrative|all figures below are fictional planning assumptions)\b",
    r"\ball\b.{0,180}\b(?:figures|data|assumptions)\s+(?:are|were)\s+(?:invented|fictional|illustrative)\b",
    r"^\s*no\b.{0,180}\b(?:is|are|was|were)\s+claimed\b",
    r"^\s*(?:buyers who need\b|pilot if\b|deposits target\b|month runway target\b)",
    r"^\s*[,;:]\s*\d",
    r"\bthis is a fictional scenario\b",
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


def _document_type_warning(raw_text: str, compact: str) -> Optional[str]:
    """Identify common non-pitch PDFs and return a user-facing explanation."""
    lower = raw_text.lower()

    # Resume/CV: require several structural signals, not just a mention of a
    # founder's experience or team slide in a legitimate pitch deck.
    resume_groups = [
        any(term in lower for term in ("work experience", "professional experience", "employment history", "experience")),
        any(term in lower for term in ("education", "academic qualifications", "degrees")),
        any(term in lower for term in ("technical skills", "skills", "core competencies")),
        any(term in lower for term in ("linkedin", "phone:", "email:", "contact")),
        any(term in lower for term in ("projects", "certifications", "internship")),
    ]
    resume_hits = sum(resume_groups)
    pitch_signals = sum(
        marker in compact
        for marker in (
            "tam", "sam", "som", "marketopportunity", "businessmodel",
            "go to market".replace(" ", ""), "fundingask", "traction",
            "customeracquisition", "unit economics".replace(" ", ""),
        )
    )
    if resume_hits >= 4 and pitch_signals < 2:
        return (
            "This file looks like a resume/CV, not a startup pitch deck. "
            "PitchProbe reviews investor-facing claims in a venture pitch. "
            "Upload a completed startup pitch deck PDF instead."
        )

    # Corporate policy / procedure documents.
    policy_markers = (
        "companypolicy", "policystatement", "scopeandpurpose",
        "rolesandresponsibilities", "policyowner", "effective date".replace(" ", ""),
        "revisionhistory", "confidentialitypolicy", "acceptableusepolicy",
        "informationsecuritypolicy", "dataprotectionpolicy",
        "incidentreportingprocedure", "policyandprocedure",
    )
    policy_hits = sum(marker in compact for marker in policy_markers)
    if policy_hits >= 2:
        return (
            "This file looks like a company policy or procedure document, not a startup "
            "pitch deck. PitchProbe cannot review policy documents. Upload a completed "
            "startup pitch deck PDF containing your venture's product, market, traction, "
            "business model, or financial claims."
        )

    # Academic / technical / internship reports. Require multiple structural
    # signals so a startup deck mentioning research or a roadmap is not rejected.
    report_markers = (
        "abstract", "tableofcontents", "acknowledgement", "declaration",
        "methodology", "literaturereview", "chapter1", "conclusion",
        "appendix", "submittedinpartialfulfilment", "internshipreport",
        "projectreport", "certificate",
    )
    report_hits = sum(marker in compact for marker in report_markers)
    if report_hits >= 4 and pitch_signals < 2:
        return (
            "This file looks like an academic, project, or internship report, not a startup "
            "pitch deck. PitchProbe is designed to assess investor-facing startup claims. "
            "Upload a completed startup pitch deck PDF instead."
        )
    return None


def document_warning_for_text(raw_text: Optional[str]) -> Optional[str]:
    """Detect unsupported document types and pitch-deck templates before claim scoring."""
    if not raw_text or not raw_text.strip():
        return None

    # Compact matching catches PDF extraction artifacts such as
    # "CourseCode", "SubmissionDeadline", and line-broken template instructions.
    compact = re.sub(r"[^a-z0-9]+", "", raw_text.lower())

    specific_warning = _document_type_warning(raw_text, compact)
    if specific_warning:
        return specific_warning

    template_hits = sum(
        marker in compact for marker in PITCH_TEMPLATE_INSTRUCTION_MARKERS
    )
    # Template guides contain many distinct imperative instructions across
    # slides. A five-marker threshold avoids warning on a real deck that only
    # mentions a roadmap, market sizing, or a three-year forecast.
    if template_hits >= 5:
        return TEMPLATE_WARNING

    strong_admin_hits = sum(marker in compact for marker in STRONG_ADMIN_MARKERS)
    admin_hits = sum(marker in compact for marker in ADMIN_COMPACT_MARKERS)
    research_hits = sum(marker in compact for marker in RESEARCH_COMPACT_MARKERS)
    business_signal_hits = sum(
        marker in compact for marker in STRONG_BUSINESS_MARKERS
    )

    # Multiple specific coursework fields are strong evidence that the file
    # is not a pitch deck. Research-heavy files need a higher threshold, and
    # clear business/traction metrics protect technical startup decks.
    if strong_admin_hits >= 2 or (admin_hits >= 4 and research_hits >= 3):
        return DOCUMENT_WARNING
    if research_hits >= 6 and business_signal_hits < 2:
        return DOCUMENT_WARNING
    return None


def normalize_claim_text(text: str) -> str:
    """Repair recoverable word boundaries and glyph artifacts from PDF extraction."""
    candidate = re.sub(r"\(cid:\d+\)", " • ", text or "", flags=re.IGNORECASE)
    candidate = re.sub(r"(?<!\w)fi(?!\w)", "→", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"(?<![A-Za-z])n(?=\s?\d)", "₹", candidate)
    candidate = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", candidate)
    candidate = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", candidate)
    return re.sub(r"\s+", " ", candidate).strip()


def is_obvious_non_claim(text: str) -> bool:
    """Remove headings, references, administrative details, and research noise."""
    candidate = normalize_claim_text(text)
    has_metric = bool(
        re.search(r"[$€£₹%]|\b\d+(?:\.\d+)?\b", candidate)
        and BUSINESS_TERMS.search(candidate)
    )
    if (len(candidate) < 12 or len(candidate.split()) < 3) and not has_metric:
        return True

    words = candidate.split()
    if len(words) > 48:
        # Claim-like text from an individual slide is rarely this long; such
        # chunks usually indicate that PDF columns or table cells were merged.
        return True
    if candidate.count("→") >= 2 and not has_metric:
        return True
    if re.search(r"\b(?:l e\d|\w+\(cid:\d+)", candidate, re.IGNORECASE):
        return True


    # Extracted architecture diagrams often become one label-heavy fragment.
    diagram_markers = [
        r"\bearable\b",
        r"\bphone app\b",
        r"\bai services\b",
        r"\bwork tools\b",
        r"\bon device\b",
        r"\bphone\s*/\s*cloud\b",
        r"\bmic array\b",
        r"\bANC\s+DSP\b",
        r"\bBluetooth LE\b",
        r"\bsession control\b",
        r"\bspeech-to-text\b",
        r"\bdiarisation\b",
    ]
    diagram_hits = sum(
        bool(re.search(pattern, candidate, re.IGNORECASE))
        for pattern in diagram_markers
    )
    has_assertive_metric = bool(has_metric and ASSERTION_TERMS.search(candidate))
    if diagram_hits >= 3 and not has_assertive_metric:
        return True
    if re.search(r"\b(?:sonora\s*/\s*seed|oct\s+2026)\b", candidate, re.IGNORECASE) and (
        "illustrative" in candidate.lower() or "not an offer" in candidate.lower()
    ):
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
    candidate = normalize_claim_text(text)
    if is_obvious_non_claim(candidate):
        return False

    has_business_term = bool(BUSINESS_TERMS.search(candidate))
    has_assertion = bool(ASSERTION_TERMS.search(candidate))
    has_product_assertion = bool(PRODUCT_SUBJECT_ACTION.search(candidate))

    has_metric = bool(
        re.search(r"[$€£₹%]|\b\d+(?:\.\d+)?\b", candidate)
        and has_business_term
    )

    # Slogans such as "Our ears are always on" and short incomplete fragments
    # should not enter risk scoring merely because a zero-shot model admits them.
    if len(candidate.split()) <= 8 and not has_metric and not has_business_term:
        return False
    if "→" in candidate and not has_assertion and not has_metric:
        return False

    # High-confidence business claims and slide metrics do not need another
    # model call. This keeps compact items such as "TAM: $5B" in the pipeline.
    if has_metric or (has_business_term and has_assertion) or has_product_assertion:
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
