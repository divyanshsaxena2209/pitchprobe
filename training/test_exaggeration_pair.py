"""Tests for group-safe train/validation splitting in the paired exaggeration experiment."""

from training.train_exaggeration_pair import LABELS, select_grouped_validation_split


def _balanced_rows():
    rows = []
    for label_index, label in enumerate(LABELS):
        for group_index in range(12):
            group = f"{label}-source-{group_index}"
            for paraphrase_index in range(2):
                rows.append({
                    "original_file_id": group,
                    "press_release_conclusion": f"press conclusion {group} {paraphrase_index}",
                    "abstract_conclusion": f"abstract conclusion {group} {paraphrase_index}",
                    "exaggeration_label": label,
                })
    return rows


def test_grouped_validation_split_has_all_labels_and_no_source_leakage():
    rows = _balanced_rows()

    train_a, validation_a = select_grouped_validation_split(rows)
    train_b, validation_b = select_grouped_validation_split(rows)

    assert train_a == train_b
    assert validation_a == validation_b
    assert set(train_a).isdisjoint(validation_a)

    train_groups = {rows[index]["original_file_id"] for index in train_a}
    validation_groups = {rows[index]["original_file_id"] for index in validation_a}
    assert train_groups.isdisjoint(validation_groups)

    assert {rows[index]["exaggeration_label"] for index in train_a} == set(LABELS)
    assert {rows[index]["exaggeration_label"] for index in validation_a} == set(LABELS)


def test_rejects_missing_source_group():
    rows = _balanced_rows()
    rows[0]["original_file_id"] = ""
    try:
        select_grouped_validation_split(rows)
    except ValueError as exc:
        assert "original_file_id" in str(exc)
    else:
        raise AssertionError("Expected missing source IDs to be rejected.")
