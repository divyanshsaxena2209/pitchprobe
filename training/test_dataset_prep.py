"""Regression tests for external-dataset deduplication and split safety."""

from training.prepare_finclaimradar import _deduplicate_exact_text


def _row(group_id, text, label):
    return {
        "group_id": group_id,
        "text": text,
        "label": label,
        "source_label": label,
        "source_context": "test fixture",
    }


def test_exact_duplicate_same_label_is_deduplicated_and_groups_are_merged():
    rows = [
        _row("group-a", "Revenue increased by 12 percent.", "LOW"),
        _row("group-b", " Revenue   increased by 12 percent. ", "LOW"),
        _row("group-a", "A distinct claim from group A.", "MODERATE"),
        _row("group-b", "A distinct claim from group B.", "HIGH"),
    ]

    result = _deduplicate_exact_text(rows)

    assert len(result) == 3
    assert len({row["text"].strip().casefold() for row in result}) == 3
    group_ids = {
        row["group_id"]
        for row in result
        if row["text"].startswith("A distinct claim")
    }
    assert len(group_ids) == 1


def test_exact_duplicate_with_conflicting_labels_is_removed():
    rows = [
        _row("group-a", "Revenue increased by 12 percent.", "LOW"),
        _row("group-b", "Revenue increased by 12 percent.", "HIGH"),
        _row("group-a", "Unique retained claim A.", "MODERATE"),
        _row("group-b", "Unique retained claim B.", "HIGH"),
    ]

    result = _deduplicate_exact_text(rows)

    assert len(result) == 2
    assert all("Revenue increased by 12 percent." != row["text"] for row in result)
    assert {row["group_id"] for row in result} == {"group-a"}
