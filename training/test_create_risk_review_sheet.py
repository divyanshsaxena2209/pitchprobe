import pandas as pd
import pytest

from training.create_risk_review_sheet import create_review_sheet


def test_audit_sheet_uses_only_train_split_and_blanks_human_labels(tmp_path):
    source = pd.DataFrame([
        {"deck_id": "a", "text": "Revenue grew 20% in six months.", "label": "LOW", "split": "train", "notes": "public"},
        {"deck_id": "b", "text": "We will dominate the entire global market.", "label": "HIGH", "split": "train", "notes": "public"},
        {"deck_id": "c", "text": "Market observation.", "label": "MODERATE", "split": "test", "notes": "test-only"},
    ])
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "review.csv"
    source.to_csv(input_csv, index=False)
    result = create_review_sheet(str(input_csv), str(output_csv), sample_size=10)
    assert len(result) == 2
    assert result["human_risk_label"].eq("").all()
    assert not result["claim_text"].str.contains("Market observation").any()
    assert "NEEDS_CONTEXT" in result.iloc[0]["risk_label_rubric"]


def test_audit_requires_split_and_label(tmp_path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Claim"]}).to_csv(input_csv, index=False)
    with pytest.raises(ValueError):
        create_review_sheet(str(input_csv), str(tmp_path / "out.csv"))
