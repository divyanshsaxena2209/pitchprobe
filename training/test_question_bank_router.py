import pandas as pd
import pytest

from training.question_bank_router import (
    InvestorQuestionRetriever,
    claim_specific_followup,
    expand_retrieval_query,
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
    examples = [
        ("Financials", "unit economics", "Which supplier quotes support the stated gross margin?", "supplier quotes; raw result; date"),
        ("Market", "market definition", "What bottom-up assumptions support this market estimate?", "eligible buyers; geography; price; adoption"),
        ("Offering", "measured value claim", "What baseline and measurement window support this customer benefit?", "baseline; measurement window; sample size; exclusions; raw result"),
        ("Competitors", "competitive moat", "What evidence shows customers would switch from competitors?", "customer interviews; procurement shortlists; alternatives"),
        ("Offering", "product capability", "Can you demonstrate the supported use cases with a product walkthrough?", "supported-use list; unsupported-use list; product walkthrough"),
        ("Financials", "pricing and gross margin", "At this seat price, which variable costs are included in gross margin?", "price per seat per month; COGS; hosting; inference; support; payment fees; usage; gross margin formula"),
    ]
    for index, (dimension, claim_type, question, evidence) in enumerate(examples, start=1):
        rows.append({
            "question_id": f"Q-{index}",
            "dimension": dimension,
            "claim_type": claim_type,
            "falsification_question": question,
            "evidence_to_request": evidence,
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
    assert 0.0 <= result[0]["claim_similarity"] <= 1.0
    assert 0.0 <= result[0]["retrieval_score"] <= 1.0
    assert result[0]["ranking_score"] >= result[0]["retrieval_score"]
    assert "query_expanded" in result[0]


def test_short_claim_rejected():
    retriever = InvestorQuestionRetriever(make_bank())
    with pytest.raises(ValueError):
        retriever.retrieve("ARR 2M")



def test_query_expansion_targets_quantified_outcomes_and_pricing():
    outcome_query = expand_retrieval_query(
        "Our product saves customer-support teams four hours per week, based on 20 pilot interviews."
    ).lower()
    pricing_query = expand_retrieval_query(
        "We charge $99 per seat per month and expect an 85% gross margin at scale."
    ).lower()

    assert "measurement window" in outcome_query
    assert "sample size" in outcome_query
    assert "cost of goods sold" in pricing_query
    assert "inference" in pricing_query


def test_measured_customer_saving_prefers_baseline_over_generic_product_walkthrough():
    retriever = InvestorQuestionRetriever(make_bank())
    result = retriever.retrieve(
        "Our product saves customer-support teams four hours per week, based on 20 pilot interviews.",
        top_k=1,
    )
    assert result[0]["claim_type"] == "measured value claim"
    assert "measurement window" in result[0]["evidence_to_request"]


def test_price_and_gross_margin_claim_prefers_cost_breakdown():
    retriever = InvestorQuestionRetriever(make_bank())
    result = retriever.retrieve(
        "We charge $99 per seat per month and expect an 85% gross margin at scale.",
        top_k=1,
    )
    assert result[0]["claim_type"] == "pricing and gross margin"
    assert any(term in result[0]["evidence_to_request"].lower() for term in ("cogs", "cost of goods sold"))


def test_claim_specific_followup_targets_quantified_customer_benefit():
    followup = claim_specific_followup(
        "Our product saves customer-support teams four hours per week, based on 20 pilot interviews."
    )
    assert "pre-product baseline" in followup
    assert "measurement window" in followup
    assert "failed cases" in followup


def test_claim_specific_followup_targets_gross_margin_costs():
    followup = claim_specific_followup(
        "We charge $99 per seat per month and expect an 85% gross margin at scale."
    )
    assert "$99" in followup
    assert "variable costs" in followup
    assert "model inference" in followup
    assert "projected at scale" in followup


def test_claim_specific_followup_targets_market_share_feasibility():
    followup = claim_specific_followup(
        "Our total addressable market is $8 billion and we expect to capture 5% within three years."
    )
    assert "eligible-buyer count" in followup
    assert "conversion rate" in followup
    assert "observed rather than projected" in followup


def test_claim_specific_followup_targets_arr_cac_and_churn_definitions():
    followup = claim_specific_followup(
        "Our annual recurring revenue is $1.2 million, customer acquisition cost is $500, and monthly churn is 3%."
    )
    assert "denominator and time window" in followup
    assert "acquisition spend" in followup
    assert "source records" in followup


def test_claim_specific_followup_targets_missing_operations_owner():
    followup = claim_specific_followup(
        "Our founder has scaled two B2B companies, but we lack an operations leader."
    )
    assert "Who owns" in followup
    assert "milestone is at risk" in followup
    assert "hiring or partner timeline" in followup


def test_claim_specific_followup_targets_supplier_capacity_and_contingency():
    followup = claim_specific_followup(
        "Our supplier can manufacture 10,000 units a month at 98% yield with current tooling."
    )
    assert "validated throughput" in followup
    assert "repeatable yield" in followup
    assert "second-source" in followup
