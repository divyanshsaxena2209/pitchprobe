"""Fast tests for paired question-generation data validation."""

import pandas as pd
import pytest

from training.train_question_generator import load_pairs, make_prompt


COLUMNS = [
    "example_id", "claim", "risk", "dimension", "context", "question",
    "split", "source_group_id", "source_name", "source_url", "license",
    "attribution", "reviewed",
]


def valid_rows():
    rows = []
    # Ensure minimum counts while keeping distinct source groups and claims.
    for i in range(20):
        rows.append([
            f"tr-{i}", f"Claim about measured customer activation {i}.",
            "Moderate", "Traction", "Pilot report available.",
            f"What cohort evidence validates activation result {i}?",
            "train", f"train-source-{i}", "Synthetic test fixture",
            "https://example.org/source", "CC0-1.0", "PitchProbe test fixture", "true",
        ])
    for split, prefix, count in [("validation", "va", 3), ("test", "te", 3)]:
        for i in range(count):
            rows.append([
                f"{prefix}-{i}", f"Distinct claim in {split} set {i}.",
                "High", "Financials", "", f"What evidence supports forecast {split} {i}?",
                split, f"{prefix}-source-{i}", "Synthetic test fixture",
                "https://example.org/source", "CC0-1.0", "PitchProbe test fixture", "true",
            ])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_make_prompt_includes_claim_risk_dimension_and_context():
    prompt = make_prompt({
        "claim": "We will reach ₹10 crore revenue.",
        "risk": "HIGH",
        "dimension": "Financials",
        "context": "No paid customers yet.",
    })
    assert "Risk category: HIGH" in prompt
    assert "Review dimension: Financials" in prompt
    assert "No paid customers yet." in prompt
    assert "We will reach ₹10 crore revenue." in prompt


def test_load_pairs_accepts_reviewed_group_disjoint_dataset(tmp_path):
    path = tmp_path / "pairs.csv"
    valid_rows().to_csv(path, index=False)
    loaded = load_pairs(path)
    assert len(loaded) == 26
    assert set(loaded["split"]) == {"train", "validation", "test"}


def test_load_pairs_rejects_source_group_leakage(tmp_path):
    frame = valid_rows()
    frame.loc[0, "source_group_id"] = frame.loc[20, "source_group_id"]
    path = tmp_path / "leak.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source_group_id occurs across multiple splits"):
        load_pairs(path)


def test_load_pairs_rejects_unreviewed_rows(tmp_path):
    frame = valid_rows()
    frame.loc[0, "reviewed"] = "false"
    path = tmp_path / "unreviewed.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Only human-reviewed rows"):
        load_pairs(path)


def test_load_pairs_rejects_duplicate_claim_across_splits(tmp_path):
    frame = valid_rows()
    frame.loc[20, "claim"] = frame.loc[0, "claim"]
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="normalized claim"):
        load_pairs(path)
