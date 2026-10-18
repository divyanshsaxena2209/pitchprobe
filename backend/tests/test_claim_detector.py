from backend.analyzer import ModelManager, analyze_sentences
from backend.claim_detector import (
    DOCUMENT_WARNING,
    document_warning_for_text,
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
