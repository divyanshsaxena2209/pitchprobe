"""Create a human-review sheet for risk-label auditing.

Input is an externally sourced FinClaimRadar CSV prepared by the existing
pipeline. This does NOT convert its source labels into gold labels. Reviewers
must independently apply the rubric before any human-reviewed examples are
used for model fitting. It should not be called a startup-pitch benchmark.
"""

import argparse
import random
from pathlib import Path

import pandas as pd

RUBRIC = {
    "LOW": "Specific bounded statement with concrete evidence/source or a past observable result; not automatically LOW just because it contains a number.",
    "MODERATE": "Plausible but underspecified claim, interpretation, market observation, or forward-looking statement whose assumptions/basis need validation.",
    "HIGH": "Material unsupported assertion, rumor, absolute/superlative or strongly overstated result without suitable evidence.",
    "NEEDS_CONTEXT": "Cannot label fairly without missing context/evidence; exclude from training unless a reviewer resolves it.",
}


def create_review_sheet(input_csv: str, output_csv: str, sample_size: int = 60, seed: int = 42) -> pd.DataFrame:
    source = pd.read_csv(input_csv)
    text_col = next((name for name in ("text", "claim_text", "claim") if name in source.columns), None)
    split_col = "split" if "split" in source.columns else None
    label_col = next((name for name in ("label", "proxy_label", "source_label") if name in source.columns), None)
    if text_col is None:
        raise ValueError("Input needs a claim text column: text, claim_text, or claim.")
    if split_col is None:
        raise ValueError("Input must contain split labels. Do not audit an unpartitioned dataset.")
    if label_col is None:
        raise ValueError("Input must retain its original proxy/source label for auditing.")

    # Sample only training rows. Do not edit/relabel validation/test examples
    # and then silently reuse them for training.
    train = source[source[split_col].astype(str).str.lower().eq("train")].copy()
    if train.empty:
        raise ValueError("No training split found.")
    rng = random.Random(seed)

    # Stratify by original mapped/proxy label to ensure we review all categories.
    groups = [
        (label, frame)
        for label, frame in train.groupby(label_col, dropna=False)
    ]
    selected = []
    base = max(1, sample_size // max(1, len(groups)))
    for _, frame in groups:
        idx = list(frame.index)
        rng.shuffle(idx)
        selected.extend(idx[:base])
    if len(selected) < sample_size:
        remaining = [idx for idx in train.index if idx not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])
    selected = selected[:sample_size]
    audit = train.loc[selected].copy().reset_index(drop=True)
    audit = audit.rename(columns={text_col: "claim_text", label_col: "original_proxy_label"})
    audit["human_risk_label"] = ""
    audit["claim_is_investor_relevant"] = ""
    audit["evidence_status"] = ""
    audit["reviewer_notes"] = ""
    audit["reviewer_id"] = ""
    audit["second_review_label"] = ""
    audit["adjudicated_label"] = ""
    audit["risk_label_rubric"] = ""
    audit["license_note"] = "Source FinClaimRadar dataset card lists CC BY-SA 4.0; verify/retain attribution and share-alike obligations."
    for label, definition in RUBRIC.items():
        audit["risk_label_rubric"] += f"{label}: {definition}\n"
    cols = [
        "claim_text",
        "original_proxy_label",
        "human_risk_label",
        "claim_is_investor_relevant",
        "evidence_status",
        "reviewer_notes",
        "reviewer_id",
        "second_review_label",
        "adjudicated_label",
        "risk_label_rubric",
        "license_note",
    ]
    if "deck_id" in audit.columns:
        cols.insert(0, "deck_id")
    if "notes" in audit.columns:
        cols.append("source_context_notes")
        audit = audit.rename(columns={"notes": "source_context_notes"})
    result = audit[cols]
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(f"Wrote {len(result)} claim-review rows to {path.resolve()}")
    print("IMPORTANT: This is a human annotation sheet, not a labelled benchmark yet.")
    print("Use only resolved human/adjudicated labels for fitting; do not use validation/test rows for training.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Prepared CSV with claim text, proxy label, and split")
    parser.add_argument("--output-csv", default="training/data/manual_risk_review.csv")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_review_sheet(args.input_csv, args.output_csv, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
