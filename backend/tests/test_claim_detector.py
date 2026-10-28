from backend.analyzer import ModelManager, analyze_sentences
from backend.claim_detector import (
    DOCUMENT_WARNING,
    TEMPLATE_WARNING,
    document_warning_for_text,\n    document_warning_for_filename,
    is_investor_claim,
)
from backend.segmenter import split_into_sentences


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
    # Many pitch slides use short quantified bullets rather than full sentences.
    assert is_investor_claim("TAM: $5B")


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


def test_detects_compacted_academic_metadata():
    compacted_text = (
        "B.TechComputerScience CourseCode BCSE306L "
        "SubmissionDeadline 07August2026 ContributionMatrix "
        "LiteratureSurvey ResearchGaps EvaluationMetrics References"
    )
    assert document_warning_for_text(compacted_text) == DOCUMENT_WARNING


def test_segmenter_keeps_compact_market_metric_bullet():
    assert "TAM: $5B" in split_into_sentences("TAM: $5B")


def test_finetuned_classifier_path_screens_claims_then_assigns_three_risk_labels(monkeypatch):
    import backend.analyzer as analyzer

    manager = ModelManager.get_instance()
    monkeypatch.setattr(manager, "classifier_mode", "fine_tuned")
    monkeypatch.setattr(manager, "classifier_loaded", True)
    monkeypatch.setattr(manager, "classifier_name", "test-three-label-checkpoint")
    monkeypatch.setattr(manager, "generator_loaded", False)

    sentences = [
        "PRODUCT OVERVIEW",
        "We achieved $10k MRR, verified by Stripe export.",
        "Our platform has significant traction.",
        "We will capture the entire trillion-dollar market with no competitors.",
    ]
    risk_labels = {
        sentences[1]: ("LOW", 0.88),
        sentences[2]: ("MODERATE", 0.77),
        sentences[3]: ("HIGH", 0.91),
    }

    monkeypatch.setattr(
        analyzer,
        "is_investor_claim",
        lambda text, manager=None: text != "PRODUCT OVERVIEW",
    )
    monkeypatch.setattr(
        analyzer,
        "classify_sentence_finetuned",
        lambda text, _manager: risk_labels[text],
    )

    results, summary, pipeline_info = analyze_sentences(
        sentences,
        raw_text="We are a startup building software for small businesses.",
    )

    assert [item["risk"] for item in results] == ["Low", "Moderate", "High"]
    assert summary == {"low_risk": 1, "moderate_risk": 1, "high_risk": 1}
    assert pipeline_info["mode"].startswith("fine_tuned_three_label")
    assert all(item["question"] is None for item in results if item["risk"] == "Low")
    assert all(item["question"] for item in results if item["risk"] in ("Moderate", "High"))



def test_cpu_device_override_does_not_probe_cuda(monkeypatch):
    import torch
    import backend.analyzer as analyzer

    monkeypatch.setenv("PITCHPROBE_DEVICE", "cpu")
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA must not be probed in CPU mode")),
    )
    manager = analyzer.ModelManager()
    assert manager.device == "cpu"


def test_excludes_incomplete_fragments_and_pitch_deck_footer_noise():
    assert not is_investor_claim("Our workflows are not.")
    assert not is_investor_claim(
        "SONORA / SEED • OCT 2026 / ILLUSTRATIVE — NOT AN OFFER 14 14 / TEAM "
        "A deliberately small founding team with clear capability gaps."
    )
    assert not is_investor_claim(
        "EARABLE PHONE APP AI SERVICES WORK TOOLS 6-mic array ANC DSP VAD "
        "touch controls Bluetooth audio handoff speech-to-text diarisation summary"
    )



def test_excludes_fictional_disclaimers_orphaned_metrics_and_nonclaims():
    assert not is_investor_claim(
        "All company, founder, waitlist, interview, pricing, unit-cost, revenue, "
        "EBITDA and adoption figures are invented for this fictional scenario."
    )
    assert not is_investor_claim(
        "No patent, regulatory approval, customer contract, revenue, paid shipment "
        "or advisor relationship is claimed."
    )
    assert not is_investor_claim(",000 paid pre-orders / refundable")
    assert not is_investor_claim("pilot if retention supports it.")
    assert not is_investor_claim("buyers who need fully offline AI or universal integrations.")


def test_detects_pitch_deck_instruction_template_and_does_not_score_prompts():
    template_text = """
    Pitch Deck Example
    The goal of your pitch deck is to give a snapshot of your investment opportunity.
    You will be allotted 10 minutes or less to present.
    Ensure slides are not text or data heavy.
    TAGLINE: Define your venture with a short declarative sentence.
    Describe the pain of the customer.
    Outline how the customer addresses the issue today.
    Demonstrate your company's value proposition.
    Provide cases, customer insights, case studies.
    Provide bottoms-up market sizing.
    Identify/profile the customer you cater to.
    Give a list of direct competitors, substitutes.
    How will you reach your target market?
    Where will you promote your product?
    Explain how you make money.
    Provide a 3 year forecast, including annual revenue.
    How much capital will you need prior to becoming cash flow positive?
    Tip: You need to convey that you are the team to execute this venture!
    Additional slides that can be used for follow-up questions.
    """
    assert document_warning_for_text(template_text) == TEMPLATE_WARNING

    results, summary, pipeline_info = analyze_sentences(
        [
            "Is it costing the customer time, money, frustration?",
            "How will you reach your target market?",
            "Provide bottoms-up market sizing.",
            "Tip: You need to convey that you are the team to execute this venture!",
        ],
        raw_text=template_text,
    )

    assert results == []
    assert summary == {"low_risk": 0, "moderate_risk": 0, "high_risk": 0}
    assert pipeline_info["document_warning"] == TEMPLATE_WARNING
    assert "No investor-facing claims were scored" in pipeline_info["document_warning"]


def test_completed_pitch_deck_is_not_flagged_for_a_few_similar_business_terms():
    completed_deck = """
    Nivaro Footwear makes walking shoes for urban commuters.
    We achieved ₹10 lakh in revenue from 250 customers during our first quarter.
    Our three-year forecast targets ₹4 crore annual revenue.
    Our go-to-market plan uses direct online sales and selected retail partners.
    """
    assert document_warning_for_text(completed_deck) is None


def test_template_detector_handles_compacted_and_line_broken_pdf_text():
    extracted_template = (
        "Pitch Deck Example Goal The goal of your pitch deck is to give a snapshot of "
        "your investment opportunity. Housekeeping You will be allotted 10 minutes or less. "
        "Cover page TAGLINE Define your venture with a short declarative sentence. "
        "Problem Describe the pain of the customer. Solution Provide cases, customer insights, case studies. "
        "Market Size Dynamic Provide bottoms-up market sizing. "
        "Go-to-Market Strategy How will you reach your target market? "
        "Financial Snapshot Provide a 3 year forecast including annual revenue. "
        "Team Tip You need to convey that you are the team to execute this venture."
    )
    assert document_warning_for_text(extracted_template) == TEMPLATE_WARNING



def test_rejects_resume_with_specific_message():
    resume_text = """
    Divyansh Saxena
    Email: divyansh@example.com | LinkedIn
    EDUCATION
    B.Tech Computer Science and Engineering
    TECHNICAL SKILLS
    Python, Java, SQL
    EXPERIENCE
    Software Engineering Intern
    PROJECTS
    Certifications
    """
    warning = document_warning_for_text(resume_text)
    assert warning is not None
    assert "resume/CV" in warning
    assert "startup pitch deck" in warning


def test_rejects_company_policy_with_specific_message():
    policy_text = """
    COMPANY POLICY
    Policy Statement
    Scope and Purpose
    Roles and Responsibilities
    Policy Owner
    Effective Date
    Revision History
    Information Security Policy
    """
    warning = document_warning_for_text(policy_text)
    assert warning is not None
    assert "policy or procedure document" in warning
    assert "startup pitch deck" in warning


def test_rejects_project_or_internship_report_with_specific_message():
    report_text = """
    Internship Report
    Declaration
    Certificate
    Acknowledgement
    Table of Contents
    Abstract
    Chapter 1 Introduction
    Methodology
    Conclusion
    Appendix
    """
    warning = document_warning_for_text(report_text)
    assert warning is not None
    assert "report" in warning
    assert "startup pitch deck" in warning


def test_completed_pitch_deck_with_team_and_product_sections_is_not_a_resume():
    deck_text = """
    Startup Pitch Deck
    Product and Market Opportunity
    TAM: $5B
    Our platform serves small businesses.
    Team: founders with experience in software and sales.
    Business Model: annual subscriptions.
    Funding Ask: $2M to reach 10,000 customers.
    """
    assert document_warning_for_text(deck_text) is None



def test_exact_reference_non_pitch_filenames_are_rejected_before_parsing():
    cases = {
        "DA1_PitchProbe_Assignment.pdf": "academic assignment",
        "CAD_Final_Report.pdf": "CAD final report",
        "Asterion_Security_Policy.pdf": "Asterion security policy",
        "BluePeak_Security_Policy.pdf": "BluePeak security policy",
        "NVC_Pitch_Deck_Template.pdf": "NVC pitch-deck template",
        "Resume.pdf": "resume/CV",
    }
    for filename, expected_message in cases.items():
        warning = document_warning_for_filename(filename)
        assert warning is not None, filename
        assert expected_message.lower() in warning.lower(), filename


def test_reference_pitch_deck_filenames_are_not_denylisted():
    for filename in (
        "Nivaro_Footwear_Pitch_Deck.pdf",
        "Sonora_Seed_Pitch_Deck.pdf",
        "StrideForge_PitchDeck.pdf",
    ):
        assert document_warning_for_filename(filename) is None
