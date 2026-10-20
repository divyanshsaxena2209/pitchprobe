import pandas as pd
import pytest

from training.question_bank_router import (
    InvestorQuestionRetriever,
    infer_dimension,
)


def test_dimension_router_covers_common_investor_claims():
    examples = {
        "We reached $2M ARR with 72% gross margin.": "Financials",
        "Our TAM is $8B across three markets.": "Market",  # TAM is market sizing, not financial performance
        "Our competitors cannot copy our patented IP moat.": "Competitors",
        "Our AI model achieves 98% accuracy in a benchmark.": "Technology & IP",
        "We are raising $3M to fund the next product milestone.": "Fundability",
        "Our product saves nurses 4 hours per week.": "Offering",
        "The founder has ten years of logistics experience.": "Team",
        "We need GDPR compliance before entering Germany.": "Legal & Regulatory",
        "Our supplier can scale production to 10,000 units.": "Scalability",
        "We expect acquisition by a strategic buyer.": "Exit",
        "The customer problem is costly and happens daily.": "Business Idea",
        "The pitch deck headline says we are the industry leader.": "Presentation",
    }
    for claim, expected in examples.items():
        assert infer_dimension(claim) == expected


def make_bank():
    rows = []
    for index, (dimension, claim_type, question) in enumerate([
        ("Financials", "unit economics", "Which supplier quotes support the stated gross margin?"),
        ("Market", "market definition", "What bottom-up assumptions support this market estimate?"),
        ("Offering", "value claim", "What baseline and measurement window support this customer benefit?"),
        ("Competitors", "competitive moat", "What evidence shows customers would switch from competitors?"),
    ], start=1):
        rows.append({
            "question_id": f"Q-{index}",
            "dimension": dimension,
            "claim_type": claim_type,
            "falsification_question": question,
            "evidence_to_request": "Raw result, sample, exclusions and date",
            "contradiction_signal": "The result disappears under an alternate assumption",
            "decision_state_if_unresolved": "open question",
            "founder_repair_action": "Add evidence and state its limit",
            "stage_note": "Label hypotheses separately from observed outcomes",
        })
    return pd.DataFrame(rows)


def test_retriever_routes_financial_claim_and_returns_evidence_fields():
    retriever = InvestorQuestionRetriever(make_bank())
    result = retriever.retrieve(
        "We reached $2M ARR and claim 80% gross margin.", top_k=2
    )
    assert result
    assert all(item["dimension"] == "Financials" for item in result)
    assert "falsification_question" in result[0]
    assert "evidence_to_request" in result[0]
    assert result[0]["dimension_route"] == "Financials"
    assert result[0]["attribution"].startswith("Playful Pixels Oy")


def test_short_claim_rejected():
    retriever = InvestorQuestionRetriever(make_bank())
    with pytest.raises(ValueError):
        retriever.retrieve("ARR 2M")
