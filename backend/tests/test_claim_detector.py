from backend.analyzer import analyze_sentences
from backend.claim_detector import (
    DOCUMENT_WARNING,
    document_warning_for_text,
    is_investor_claim,
)


def test_keeps_investor_facing_business_claims():
    assert is_investor_claim(
        "We achieved $10k MRR with 12% month-on-month growth, verified by Stripe export."
    )
    assert is_investor_claim(
        "Our platform reduces invoice processing time by 70% for small businesses."
    )
    assert is_investor_claim(
        "We signed 18 paid pilots and expect to convert 60% into annual contracts."
    )


def test_excludes_headings_references_and_background_text():
    assert not is_investor_claim("PRODUCT OVERVIEW")
    assert not is_investor_claim("References")
    assert not is_investor_claim(
        '[1] A. Gupta and R. Sharma, "Claim Plausibility Scoring", IEEE Access, 2024.'
    )
    assert not is_investor_claim(
        "The primary stakeholder is a first-time founder preparing for a seed or Series-A pitch."
    )


def test_identifies_academic_assignment_document():
    academic_text = """
    Vellore Institute of Technology, Chennai
    B.Tech Computer Science and Engineering
    Course Code: BCSE306L
    Assignment DA1
    Submission Deadline: 07 August 2026
    Literature Survey
    Research Gaps
    Proposed System Architecture
    Evaluation Metrics
    Contribution Matrix
    References
    """
    warning = document_warning_for_text(academic_text)
    assert warning == DOCUMENT_WARNING


def test_analyzer_skips_non_pitch_document_instead_of_scoring_every_sentence():
    academic_text = """
    B.Tech Computer Science and Engineering
    Course Code: BCSE306L
    Submission Deadline: 07 August 2026
    Literature Survey
    Research Gaps
    Proposed System Architecture
    Contribution Matrix
    """
    sentences = [
        "LITERATURE SURVEY",
        "The primary stakeholder is a first-time founder preparing for a seed round.",
        "Papers that perform question generation do not condition generation on an upstream classifier.",
    ]

    results, summary, pipeline_info = analyze_sentences(
        sentences,
        raw_text=academic_text,
    )

    assert results == []
    assert summary == {"low_risk": 0, "moderate_risk": 0, "high_risk": 0}
    assert pipeline_info["document_warning"] == DOCUMENT_WARNING


def test_analyzer_scores_claims_but_skips_non_claims():
    sentences = [
        "PRODUCT OVERVIEW",
        "References",
        '[1] A. Gupta, "Claim Plausibility Scoring", IEEE Access, 2024.',
        "We achieved $10k MRR with 12% month-on-month growth, verified by Stripe export.",
        "The primary stakeholder is a first-time founder preparing for seed funding.",
    ]

    results, summary, pipeline_info = analyze_sentences(
        sentences,
        raw_text="We are a startup selling software to small businesses.",
    )

    assert len(results) == 1
    assert results[0]["text"].startswith("We achieved $10k MRR")
    assert summary["low_risk"] + summary["moderate_risk"] + summary["high_risk"] == 1
    assert pipeline_info["document_warning"] is None
