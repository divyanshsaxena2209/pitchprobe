from types import SimpleNamespace

import torch

from backend.analyzer import (
    analyze_sentences,
    classify_sentence_nli,
    generate_objection_question,
    _is_useful_generated_question,
    _targeted_investor_question,
    _question_key,
    _question_similarity,
    ModelManager,
)


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



def test_irrelevant_flan_questions_are_rejected():
    assert not _is_useful_generated_question("What is the best title for the article?")
    assert not _is_useful_generated_question("What is a tough VC objection?")
    assert _is_useful_generated_question(
        "Which bottom-up assumptions support this market-size estimate?"
    )


def test_question_templates_are_specific_to_claim_topic():
    market_question = _targeted_investor_question(
        "Our SAM is a management assumption and has not been validated.",
        "Moderate",
    )
    assert "customer segment" in market_question
    assert "market estimate" in market_question

    traction_question = _targeted_investor_question(
        "We have 310 waitlist sign-ups, but zero paid units shipped.",
        "High",
    )
    assert "paying customers" in traction_question
    assert "conversion rate" in traction_question

    financial_question = _targeted_investor_question(
        "We forecast ₹42 crore revenue in year three if conversion holds.",
        "High",
    )
    assert "gross margin" in financial_question
    assert "forecast" in financial_question


def test_flan_t5_is_opt_in_by_default(monkeypatch):
    monkeypatch.delenv("PITCHPROBE_USE_FLAN_T5", raising=False)
    manager = ModelManager()
    assert manager.use_flan_t5 is False



def test_question_routing_distinguishes_icp_pricing_and_team():
    icp = _targeted_investor_question(
        "customer-facing teams with 8+ hours of weekly calls.",
        "Moderate",
    )
    assert "target segment" in icp
    assert "willingness to pay" in icp

    pricing = _targeted_investor_question(
        "₹499 / month Unlimited meeting minutes within fair-use limits, richer summaries and team templates.",
        "Moderate",
    )
    assert "price tier" in pricing
    assert "service costs" in pricing

    hiring = _targeted_investor_question(
        "We need a fractional operations / supply-chain leader, privacy counsel and independent acoustic testing.",
        "Moderate",
    )
    assert "capability" in hiring
    assert "hiring" in hiring


def test_fundraise_and_connectivity_claims_get_relevant_questions():
    fundraise = _targeted_investor_question(
        "Raising 8 crore to reach a validated, shippable product.",
        "High",
    )
    assert "milestones" in fundraise
    assert "runway" in fundraise

    connectivity = _targeted_investor_question(
        "Cloud AI is optional for audio playback; meeting intelligence requires app connectivity and user consent.",
        "Moderate",
    )
    assert "connectivity" in connectivity
    assert "consent" in connectivity


def test_question_generation_varies_wording_and_never_repeats_within_a_deck():
    manager = ModelManager()
    manager.use_flan_t5 = False
    similar_claims = [
        "We have 310 waitlist sign-ups, but zero paid units shipped.",
        "Our waitlist has 420 sign-ups and no paid units shipped yet.",
        "The beta attracted 200 users, but conversion to paying customers remains low.",
        "We have strong traction among enterprise users, but retention is unproven.",
        "Our pilot participants show growing activation but weak retention.",
        "We acquired 500 users, but only a small fraction became paying customers.",
        "Customer interviews show interest, but few users return after their first session.",
        "We are rapidly growing and converting many customers after free trials.",
    ]

    questions = []
    for claim in similar_claims:
        question = generate_objection_question(
            claim,
            "Moderate",
            manager,
            previous_questions=questions,
        )
        assert question is not None
        assert question.endswith("?")
        assert _question_key(question) not in {_question_key(item) for item in questions}
        questions.append(question)

    # The wording should differ structurally, not merely by a changed token.
    for index, question in enumerate(questions):
        for earlier in questions[:index]:
            assert _question_similarity(question, earlier) < 0.92
