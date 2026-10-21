"""Retrieval baseline for targeted investor questions.

Loads the 48-question Startup Investor Falsification Question Bank (CC BY 4.0)
and retrieves claim-relevant questions without training a generator. This is a
research prototype; it does not verify claims or replace human review.

Attribution: Playful Pixels Oy, Startup Investor Falsification Question Bank,
https://huggingface.co/datasets/mheilimo/startup-investor-falsification-question-bank
License: CC BY 4.0. Dataset pinned to commit f39e258288f16c6f85dc40e67d38b9e90eff1426.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "mheilimo/startup-investor-falsification-question-bank/resolve/"
    "f39e258288f16c6f85dc40e67d38b9e90eff1426/"
    "startup-investor-falsification-question-bank.csv"
)
REQUIRED_COLUMNS = [
    "question_id",
    "dimension",
    "claim_type",
    "falsification_question",
    "evidence_to_request",
    "contradiction_signal",
    "decision_state_if_unresolved",
    "founder_repair_action",
    "stage_note",
]

# Route first, then rank within a dimension. This prevents a claim such as
# "we reached $2M ARR" being matched only because another question mentions
# "customer value" or "market" in its evidence fields.
DIMENSION_RULES = [
    ("Fundability", r"\b(?:fundrais(?:e|ing)|rais(?:e|ing)\s+(?:\$|\d|capital|funds|a)|raising capital|funding ask|funding|investment ask|series [a-d]|use of funds|next round|runway from this raise)\b"),
    ("Financials", r"\b(?:revenue|arr|mrr|cac|ltv|gross margin|unit economics|burn rate|runway|valuation|ebitda|profit|pricing|price tier|payback|forecast|financial plan|cost per|conversion rate|churn|renewal|arpu)\b"),
    ("Market", r"\b(?:tam|sam|som|market size|market share|market estimate|market opportunity|addressable market|beachhead|market penetration|target segment|demand|buyer demand)\b"),
    ("Competitors", r"\b(?:competitors?|competition|incumbents?|competitive advantage|moat|market leader|switching costs|copy|copied|bundl(?:e|ing))\b"),
    ("Technology & IP", r"\b(?:patent|intellectual property|\bip\b|proprietary|algorithm|model accuracy|latency|benchmark|technical performance|cybersecurity|encryption|uptime|reliability)\b"),
    ("Legal & Regulatory", r"\b(?:compliance|regulatory|regulation|license|licence|hipaa|gdpr|privacy|consent|data retention|deletion policy|security audit|legal risk)\b"),
    ("Team", r"\b(?:founder|co-founder|founding team|leadership|hiring|hire|team experience|advisor|capability gap|key-person risk|execution track record)\b"),
    ("Scalability", r"\b(?:scale production|scalability|manufacturing|supplier|yield|inventory|tooling|fulfilment|fulfillment|capacity|infrastructure|production volume|geographic expansion)\b"),
    ("Exit", r"\b(?:exit strategy|acquisition|acquirer|ipo|exit multiple|strategic buyer|liquidity event)\b"),
    ("Offering", r"\b(?:product|platform|solution|feature|workflow|prototype|adoption|activation|retention|pilot|beta|user outcome|time saved|cost savings|performance|accuracy|reduces?|improves?|automates?|delivers?|customer value|willingness to pay)\b"),
    ("Business Idea", r"\b(?:problem|pain point|customer need|current workaround|buyer|user need|problem frequency|problem cost|customer identity|why now|urgency)\b"),
    ("Presentation", r"\b(?:slide|deck|headline|pitch narrative|positioning|brand promise|tagline|presentation)\b"),
]


def infer_dimension(claim: str) -> Optional[str]:
    """Infer a review dimension using transparent keywords; return None if unclear."""
    text = re.sub(r"\s+", " ", str(claim or "")).lower()
    for dimension, pattern in DIMENSION_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return dimension
    return None


def expand_retrieval_query(claim: str) -> str:
    """Add narrow retrieval hints for measured benefits and price/unit-economics claims.

    These terms affect ranking only. The original claim is preserved in every
    returned result, and the added terms are not treated as evidence.
    """
    text = re.sub(r"\s+", " ", str(claim or "")).lower()
    expansions = []

    benefit_claim = re.search(
        r"\b(?:save|saves|saved|saving|reduce|reduces|reduced|improve|improves|"
        r"improved|increase|increases|increased|faster|time saved|cost savings|"
        r"productivity|customer outcome)\b",
        text,
    )
    quantified_outcome = re.search(
        r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|twenty|thirty|forty|fifty)\s*(?:%|percent|hours?|minutes?|"
        r"seconds?|days?|weeks?|months?|dollars?|customers?|users?)\b",
        text,
    )
    if benefit_claim and quantified_outcome:
        expansions.append(
            "measured customer outcome benefit savings time saved baseline comparison "
            "before after measurement window sample size exclusions raw data self-reported "
            "survey pilot cohort control group"
        )

    pricing_or_margin = re.search(
        r"(?:[$₹€£]\s*\d|\bper\s+(?:seat|user|month|year)\b|"
        r"\b(?:pricing|price tier|gross margin|unit economics|contribution margin|"
        r"cost of goods sold|cogs|cost per seat|subscription)\b)",
        text,
    )
    if pricing_or_margin:
        expansions.append(
            "pricing seat subscription recurring monthly price tier list price realized price "
            "gross margin variable cost cost of goods sold COGS cost per seat hosting cloud "
            "infrastructure inference support payment processing discounts usage assumptions "
            "contribution margin cost breakdown"
        )

    return " ".join([str(claim or "")] + expansions)


def claim_specific_followup(claim: str) -> str:
    """Add a transparent local follow-up when retrieval misses a key claim detail.

    This is locally authored guidance, not a question copied from the licensed
    source bank. It supplements rather than replaces the attributed retrieved item.
    """
    text = re.sub(r"\s+", " ", str(claim or "")).strip()
    lowered = text.lower()

    benefit_claim = re.search(
        r"\b(?:save|saves|saved|saving|reduce|reduces|reduced|improve|improves|"
        r"improved|increase|increases|increased|faster|time saved|cost savings|"
        r"productivity|customer outcome)\b",
        lowered,
    )
    quantified_outcome = re.search(
        r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|twenty|thirty|forty|fifty)\s*(?:%|percent|hours?|minutes?|"
        r"seconds?|days?|weeks?|months?|dollars?|customers?|users?)\b",
        lowered,
    )
    if benefit_claim and quantified_outcome:
        return (
            "What pre-product baseline, measurement window and sample support the "
            "claimed customer benefit, and were setup time, failed cases and "
            "outlier users included in the calculation?"
        )

    pricing_or_margin = re.search(
        r"(?:[$₹€£]\s*\d|\bper\s+(?:seat|user|month|year)\b|"
        r"\b(?:pricing|price tier|gross margin|unit economics|contribution margin|"
        r"cost of goods sold|cogs|cost per seat|subscription)\b)",
        lowered,
    )
    if pricing_or_margin and re.search(r"\b(?:gross margin|margin|cogs|cost of goods sold)\b", lowered):
        price_match = re.search(r"[$₹€£]\s?\d[\d,.]*", text)
        price_text = f"At {price_match.group(0)}" if price_match else "At the stated price"
        return (
            f"{price_text}, which variable costs (hosting, model inference, support, "
            "payment fees, discounts and refunds) are included in the claimed gross "
            "margin, and is that margin measured today or projected at scale?"
        )

    return ""


class InvestorQuestionRetriever:
    def __init__(self, bank: pd.DataFrame):
        missing = sorted(set(REQUIRED_COLUMNS) - set(bank.columns))
        if missing:
            raise ValueError(f"Question bank is missing columns: {missing}")
        if bank.empty:
            raise ValueError("Question bank is empty")

        self.bank = bank[REQUIRED_COLUMNS].fillna("").copy()
        self.bank["dimension"] = self.bank["dimension"].astype(str)
        self.bank["claim_type"] = self.bank["claim_type"].astype(str)
        # The retrieval text weights the tested topic and evidence request, not
        # only the general wording of the question.
        self.documents = (
            self.bank["dimension"] + " " + self.bank["dimension"] + " "
            + self.bank["claim_type"] + " " + self.bank["claim_type"] + " "
            + self.bank["falsification_question"] + " "
            + self.bank["evidence_to_request"] + " "
            + self.bank["contradiction_signal"]
        ).tolist()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(self.documents)

    @classmethod
    def from_csv(cls, path: str = DATASET_URL) -> "InvestorQuestionRetriever":
        """Load a local CSV or the pinned public source URL."""
        return cls(pd.read_csv(path))

    def retrieve(self, claim: str, top_k: int = 3) -> List[Dict[str, object]]:
        """Return relevant question-bank rows for one claim, with transparent scores."""
        claim = re.sub(r"\s+", " ", str(claim or "")).strip()
        if len(claim) < 8:
            raise ValueError("Provide the full claim text, not a short label.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        # Preserve the raw claim-only score for diagnostics, but rank with a
        # narrowly expanded query when a numeric customer outcome or pricing claim
        # needs evidence terms that often do not appear verbatim in the claim.
        query = self.vectorizer.transform([claim])
        similarities = cosine_similarity(query, self.matrix).ravel()
        ranking_query_text = expand_retrieval_query(claim)
        ranking_query = self.vectorizer.transform([ranking_query_text])
        ranking_similarities = cosine_similarity(ranking_query, self.matrix).ravel()
        routed = infer_dimension(claim)
        scores = ranking_similarities.copy()

        if routed:
            in_dimension = self.bank["dimension"].eq(routed).to_numpy()
            # Strong preference, not a fabricated semantic score. If no row
            # in the routed dimension exists, fall back to unrestricted retrieval.
            if in_dimension.any():
                scores = scores + in_dimension.astype(float) * 0.35
                candidates = [int(i) for i in range(len(self.bank)) if in_dimension[i]]
            else:
                candidates = list(range(len(self.bank)))
        else:
            candidates = list(range(len(self.bank)))

        # Stable sorting means results remain reproducible for equal ranking scores.
        ranked = sorted(candidates, key=lambda i: (-float(scores[i]), i))[:top_k]
        results: List[Dict[str, object]] = []
        for index in ranked:
            row = self.bank.iloc[index].to_dict()
            # Expose both the raw-claim and expanded-query similarities.
            # The retrieval score reflects the score used for semantic ranking;
            # the route bonus is separate so the fields remain interpretable.
            row["claim_similarity"] = round(float(similarities[index]), 4)
            row["retrieval_score"] = round(float(ranking_similarities[index]), 4)
            row["dimension_route_bonus"] = 0.35 if routed and row["dimension"] == routed else 0.0
            row["ranking_score"] = round(float(scores[index]), 4)
            row["query_expanded"] = ranking_query_text != claim
            row["dimension_route"] = routed
            row["attribution"] = "Playful Pixels Oy — Startup Investor Falsification Question Bank (CC BY 4.0)"
            row["question_for_claim"] = (
                f"For this claim — “{claim}” — {row['falsification_question'][0].lower()}"
                f"{row['falsification_question'][1:]}"
            )
            row["claim_specific_followup"] = claim_specific_followup(claim)
            results.append(row)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve targeted investor questions from an attributed public question bank."
    )
    parser.add_argument("--bank-csv", default=DATASET_URL, help="Local CSV or pinned public source URL")
    parser.add_argument("--claims-csv", required=True, help="CSV with a 'claim' column")
    parser.add_argument("--output-csv", default="training/artifacts/question_bank_matches.csv")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    claims = pd.read_csv(args.claims_csv)
    if "claim" not in claims.columns:
        raise ValueError("Input CSV must contain a 'claim' column.")
    retriever = InvestorQuestionRetriever.from_csv(args.bank_csv)

    out_rows = []
    for claim in claims["claim"].fillna("").astype(str):
        for result in retriever.retrieve(claim, top_k=args.top_k):
            out_rows.append({"claim": claim, **result})

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(output, index=False)
    print(f"Claims processed: {len(claims)}")
    print(f"Question matches written: {len(out_rows)}")
    print(f"Output: {output.resolve()}")
    print("Prototype only: review question relevance; a retrieved question does not verify the claim.")


if __name__ == "__main__":
    main()
