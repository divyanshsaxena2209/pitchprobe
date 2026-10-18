from types import SimpleNamespace

import torch

from backend.analyzer import analyze_sentences, classify_sentence_nli, generate_objection_question, ModelManager


def test_risk_mapping():
    manager = ModelManager.get_instance()

    # Grounded claim
    grounded_sentence = "We achieved $10k MRR with 12% month-on-month growth, verified by Stripe export."
    risk_low, score_low = classify_sentence_nli(grounded_sentence, manager)
    assert risk_low == "Low"
    assert 0.0 <= score_low <= 1.0

    # Hyperbolic claim
    hyperbolic_sentence = "We will capture a massive, trillion-dollar global market with no competitors."
    risk_high, score_high = classify_sentence_nli(hyperbolic_sentence, manager)
    assert risk_high == "High"
    assert 0.0 <= score_high <= 1.0

    # Vague claim
    vague_sentence = "Our platform delivers industry-leading next-gen capabilities and strong traction."
    risk_mod, score_mod = classify_sentence_nli(vague_sentence, manager)
    assert risk_mod == "Moderate"
    assert 0.0 <= score_mod <= 1.0


def test_question_generation_gating():
    manager = ModelManager.get_instance()

    # Requirement 5: Low-risk sentences MUST get NO question
    low_question = generate_objection_question("We achieved $10k MRR with 12% MoM growth.", "Low", manager)
    assert low_question is None

    # Moderate and High risk sentences MUST get exactly one investor objection question
    mod_question = generate_objection_question("We have significant traction in the enterprise sector.", "Moderate", manager)
    assert mod_question is not None
    assert isinstance(mod_question, str)
    assert len(mod_question) > 10
    assert mod_question.endswith("?")

    high_question = generate_objection_question("Our platform will easily conquer the entire trillion dollar market.", "High", manager)
    assert high_question is not None
    assert isinstance(high_question, str)
    assert len(high_question) > 10
    assert high_question.endswith("?")


def test_analyze_sentences_integration():
    sentences = [
        "We achieved $10k MRR with 12% month-on-month growth, verified by Stripe export.",
        "We will capture the entire trillion-dollar global market with no competitors.",
        "Our solution has significant traction and next-gen capabilities.",
    ]
    results, summary, pipeline_info = analyze_sentences(sentences)

    assert len(results) == 3
    assert summary["total_sentences"] if "total_sentences" in summary else (
        summary["low_risk"] + summary["moderate_risk"] + summary["high_risk"] == 3
    )
    assert summary["low_risk"] >= 1
    assert summary["high_risk"] >= 1
    assert summary["moderate_risk"] >= 1

    # Verify each sentence has expected structure and gating
    for item in results:
        assert item["risk"] in ("Low", "Moderate", "High")
        assert 0.0 <= item["score"] <= 1.0
        if item["risk"] == "Low":
            assert item["question"] is None, f"Expected no question for Low risk claim, got: {item['question']}"
        else:
            assert item["question"] is not None, f"Expected counter-question for {item['risk']} claim"
            assert item["question"].endswith("?")


def test_nli_risk_score_uses_raw_entailment_logits(monkeypatch):
    manager = ModelManager.get_instance()

    class FakeFeatures(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return FakeFeatures({"input_ids": torch.ones((3, 2), dtype=torch.long)})

    class FakeModel:
        config = SimpleNamespace(
            id2label={
                0: "contradiction",
                1: "entailment",
                2: "neutral",
            }
        )

        def __call__(self, **_kwargs):
            # Entailment logits differ only slightly. Normalizing already
            # softmaxed per-row probabilities would compress these scores
            # close to 0.333; correct cross-hypothesis softmax is more distinct.
            return SimpleNamespace(
                logits=torch.tensor(
                    [
                        [0.0, 0.3, 0.0],
                        [0.0, 0.2, 0.0],
                        [0.0, 0.1, 0.0],
                    ]
                )
            )

    monkeypatch.setattr(manager, "classifier_loaded", True)
    monkeypatch.setattr(manager, "classifier_tokenizer", FakeTokenizer())
    monkeypatch.setattr(manager, "classifier_model", FakeModel())
    monkeypatch.setattr(manager, "device", "cpu")

    risk, score = classify_sentence_nli("Example investor-facing claim.", manager)
    expected = torch.softmax(torch.tensor([0.3, 0.2, 0.1]), dim=0)[0].item()

    assert risk == "Low"
    assert abs(score - round(expected, 4)) < 1e-4
    assert score > 0.35
