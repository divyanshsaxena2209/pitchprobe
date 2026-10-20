"""Dataset loading and validation shared by baseline and fine-tuning scripts."""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

LABELS = ["LOW", "MODERATE", "HIGH"]
SPLITS = {"train", "validation", "test"}
REQUIRED_COLUMNS = {"deck_id", "text", "label", "split"}
OPTIONAL_COLUMNS = {"notes"}


def normalize_label(value: str) -> str:
    """Normalize common variants to the three canonical claim-risk labels."""
    candidate = re.sub(r"[\s-]+", "_", (value or "").strip().upper())
    aliases = {
        "GROUNDED": "LOW",
        "LOW_RISK": "LOW",
        "MODERATE_RISK": "MODERATE",
        "VAGUE": "MODERATE",
        "HIGH_RISK": "HIGH",
        "HYPERBOLIC": "HIGH",
    }
    return aliases.get(candidate, candidate)


def read_and_validate_dataset(
    path: str,
    *,
    require_all_training_labels: bool = True,
) -> List[Dict[str, str]]:
    """Load a CSV and reject errors that could invalidate the evaluation."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. For external-only training, run "
            "'python -m training.prepare_finclaimradar' first, or supply a "
            "CSV of externally sourced and consistently mapped claim labels."
        )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {', '.join(sorted(missing))}"
            )
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items() if key}
            row["label"] = normalize_label(row.get("label", ""))
            row["split"] = row.get("split", "").lower()
            row["_line"] = str(line_number)
            if not row.get("deck_id") or not row.get("text"):
                raise ValueError(
                    f"Line {line_number}: deck_id and text must not be empty."
                )
            if row["label"] not in LABELS:
                raise ValueError(
                    f"Line {line_number}: invalid label {row['label']!r}. "
                    f"Use one of {', '.join(LABELS)}."
                )
            if row["split"] not in SPLITS:
                raise ValueError(
                    f"Line {line_number}: split must be train, validation, or test."
                )
            rows.append(row)

    if not rows:
        raise ValueError(
            f"{csv_path} has no labeled rows. Add human-reviewed examples first."
        )

    decks_by_split = defaultdict(set)
    split_by_deck = {}
    normalized_text_seen = {}
    counts = Counter()
    for row in rows:
        deck_id, split = row["deck_id"], row["split"]
        previous = split_by_deck.get(deck_id)
        if previous and previous != split:
            raise ValueError(
                f"Data leakage risk: deck_id {deck_id!r} appears in both "
                f"{previous!r} and {split!r}. Keep every row from one deck in a "
                "single split."
            )
        split_by_deck[deck_id] = split
        decks_by_split[split].add(deck_id)
        normalized_text = re.sub(r"\s+", " ", row["text"]).strip().casefold()
        if normalized_text in normalized_text_seen:
            first = normalized_text_seen[normalized_text]
            raise ValueError(
                f"Duplicate normalized claim text near CSV line {row['_line']}; "
                f"first occurrence was line {first['_line']} in split {first['split']!r}. "
                "Deduplicate exact inputs before splitting to prevent overweighting "
                "and train/validation/test leakage."
            )
        normalized_text_seen[normalized_text] = row
        counts[(split, row["label"])] += 1

    for required_split in ("train", "validation", "test"):
        if not any(row["split"] == required_split for row in rows):
            raise ValueError(f"No rows found for the {required_split!r} split.")

    train_labels = {row["label"] for row in rows if row["split"] == "train"}
    missing_train_labels = set(LABELS) - train_labels
    if require_all_training_labels and missing_train_labels:
        raise ValueError(
            "The training split must contain at least one example from each label. "
            f"Missing: {', '.join(sorted(missing_train_labels))}."
        )

    print(f"Validated {len(rows)} labeled examples in {len(split_by_deck)} groups.")
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        label_counts = Counter(row["label"] for row in split_rows)
        print(
            f"  {split:10s}: {len(split_rows):4d} rows, "
            f"{len(decks_by_split[split]):3d} decks | "
            + ", ".join(f"{label}={label_counts.get(label, 0)}" for label in LABELS)
        )
    print("Group leakage check: passed (each group belongs to exactly one split).")
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate PitchProbe labeled claim data.")
    parser.add_argument("--data-csv", default="training/data/claims.csv")
    args = parser.parse_args()
    read_and_validate_dataset(args.data_csv)
