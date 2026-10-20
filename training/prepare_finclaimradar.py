"""Prepare a public-only risk-label CSV from FinClaimRadar.

This script uses only the public Hugging Face FinClaimRadar dataset. It never
reads user uploads or PitchProbe output files.

Run:
python -m training.prepare_finclaimradar --output-csv training/data/external_claims.csv

The source dataset uses evidentiary-basis categories, not PitchProbe risk labels.
The explicit mapping below is a proxy and must be evaluated before deployment.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit

from training.dataset_utils import LABELS, read_and_validate_dataset

DATASET_ID = "ImBinarry/FinClaimRadar-Financial-Reasoning"
SEED = 42

# Explicit task mapping: not an official equivalence supplied by the source dataset.
# RFD represents sourced, specific historical data; GMO/AEOA/FLS are more general,
# interpretive, or forward-looking; SSR/UAH are speculative or hype-oriented.
EVIDENCE_TO_RISK = {
    "REPORTED FACTUAL DATA": "LOW",
    "GENERAL MARKET OBSERVATION/TREND DESCRIPTION": "MODERATE",
    "ATTRIBUTED EXPERT OPINION/ANALYSIS": "MODERATE",
    "COMPANY FORWARD-LOOKING STATEMENT": "MODERATE",
    "SPECULATIVE STATEMENT/RUMOR": "HIGH",
    "UNSUPPORTED ASSERTION/HYPE": "HIGH",
}


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _get_field(row: Dict[str, Any], *candidates: str) -> str:
    by_normalized_key = {_normalized_key(key): value for key, value in row.items()}
    for candidate in candidates:
        value = by_normalized_key.get(_normalized_key(candidate))
        if value is not None and str(value).strip() and str(value).strip().lower() != "none":
            return str(value).strip()
    return ""


def _map_label(source_label: str) -> str:
    normalized = re.sub(r"\s+", " ", source_label.strip().upper())
    try:
        return EVIDENCE_TO_RISK[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unrecognised FinClaimRadar evidence label {source_label!r}; "
            f"known labels: {', '.join(EVIDENCE_TO_RISK)}"
        ) from exc


def _deduplicate_exact_text(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Deduplicate identical inputs and exclude texts with conflicting labels.

    If duplicate text occurs in different source groups, merge those groups
    before splitting so related material cannot leak across train/validation/test.
    """
    by_text: Dict[str, List[Dict[str, str]]] = {}
    parents: Dict[str, str] = {}

    def find(group_id: str) -> str:
        parents.setdefault(group_id, group_id)
        while parents[group_id] != group_id:
            parents[group_id] = parents[parents[group_id]]
            group_id = parents[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Deterministic root to make output repeatable across runs.
        first, second = sorted((left_root, right_root))
        parents[second] = first

    for row in rows:
        key = re.sub(r"\s+", " ", row["text"]).strip().casefold()
        by_text.setdefault(key, []).append(row)
        find(row["group_id"])

    for duplicates in by_text.values():
        if len(duplicates) > 1:
            base_group = duplicates[0]["group_id"]
            for duplicate in duplicates[1:]:
                union(base_group, duplicate["group_id"])

    # Resolve group roots after all unions, including chains of duplicated text.
    for row in rows:
        row["group_id"] = find(row["group_id"])

    deduplicated: List[Dict[str, str]] = []
    duplicate_rows_removed = 0
    conflicting_texts_dropped = 0

    for key, duplicates in by_text.items():
        labels = {row["label"] for row in duplicates}
        if len(labels) > 1:
            # One text cannot supervise a single-input classifier with two labels.
            conflicting_texts_dropped += 1
            duplicate_rows_removed += len(duplicates)
            continue

        # Keep one instance of exact text; all source groups have already been
        # merged so any remaining rows from those groups stay in a single split.
        chosen = dict(duplicates[0])
        chosen["group_id"] = find(chosen["group_id"])
        deduplicated.append(chosen)
        duplicate_rows_removed += len(duplicates) - 1

    print(
        f"Exact-text deduplication: removed {duplicate_rows_removed} duplicate/ambiguous rows; "
        f"dropped {conflicting_texts_dropped} text(s) with conflicting risk labels."
    )
    return deduplicated


def _merge_near_duplicate_groups(
    rows: List[Dict[str, str]],
    threshold: float = 0.70,
) -> List[Dict[str, str]]:
    """Keep highly similar claims in the same split to reduce template leakage.

    This conservative character n-gram cosine threshold is an audit heuristic,
    not a semantic-equivalence label. It groups examples for splitting but does
    not delete them or change their labels.
    """
    if len(rows) < 2:
        return rows

    texts = [re.sub(r"\s+", " ", row["text"]).strip() for row in rows]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=50000,
    )
    matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(matrix, dense_output=True)

    parents: Dict[str, str] = {}

    def find(group_id: str) -> str:
        parents.setdefault(group_id, group_id)
        while parents[group_id] != group_id:
            parents[group_id] = parents[parents[group_id]]
            group_id = parents[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parents[second] = first

    for row in rows:
        find(row["group_id"])

    near_duplicate_pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if float(similarities[i, j]) >= threshold:
                near_duplicate_pairs += 1
                union(rows[i]["group_id"], rows[j]["group_id"])

    original_group_count = len({row["group_id"] for row in rows})
    for row in rows:
        row["group_id"] = find(row["group_id"])
    final_group_count = len({row["group_id"] for row in rows})

    print(
        f"Near-duplicate grouping: {near_duplicate_pairs} pair(s) at character "
        f"TF-IDF cosine >= {threshold:.2f}; source groups reduced from "
        f"{original_group_count} to {final_group_count} split groups."
    )
    return rows


def _split_by_group(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Create reproducible 80/10/10 splits without separating related groups."""
    groups = [row["group_id"] for row in rows]
    indices = list(range(len(rows)))
    first_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=SEED
    )
    train_idx, remainder_idx = next(first_split.split(indices, groups=groups))

    remainder_rows = [rows[index] for index in remainder_idx]
    remainder_groups = [row["group_id"] for row in remainder_rows]
    second_split = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=SEED + 1
    )
    validation_local, test_local = next(
        second_split.split(
            list(range(len(remainder_rows))),
            groups=remainder_groups,
        )
    )

    split_by_group = {}
    for index in train_idx:
        split_by_group[rows[index]["group_id"]] = "train"
    for index in validation_local:
        split_by_group[remainder_rows[index]["group_id"]] = "validation"
    for index in test_local:
        split_by_group[remainder_rows[index]["group_id"]] = "test"

    output = []
    for row in rows:
        output.append(
            {
                "deck_id": f"finclaimradar::{row['group_id']}",
                "text": row["text"],
                "label": row["label"],
                "split": split_by_group[row["group_id"]],
                "notes": (
                    "External FinClaimRadar dataset; "
                    f"source_evidence_label={row['source_label']}; "
                    f"context={row['source_context'] or 'not provided'}. "
                    "Mapped proxy label; review source-category/risk alignment."
                ),
            }
        )
    return output


def prepare_dataset(output_csv: str) -> Tuple[int, Dict[str, int]]:
    print(f"Downloading public dataset: {DATASET_ID}")
    source = load_dataset(DATASET_ID, split="train")
    mapped_rows = []

    for index, raw in enumerate(source):
        row = dict(raw)
        claim_text = _get_field(row, "claim_text", "claim text")
        source_label = _get_field(
            row,
            "evidentiary_basis_classification",
            "evidentiary basis classification",
        )
        claim_id = _get_field(row, "claim_id", "claim id") or f"row_{index:04d}"
        source_context = _get_field(row, "source_context_type", "source context type")
        counterfactual_group = _get_field(
            row,
            "counterfactual_group_id",
            "Counterfactual Group ID",
        )
        if not claim_text:
            continue
        if not source_label:
            raise ValueError(
                f"Source row {claim_id!r} has no evidentiary_basis_classification."
            )

        group_id = (
            f"counterfactual_{counterfactual_group}"
            if counterfactual_group
            else f"claim_{claim_id}"
        )
        mapped_rows.append(
            {
                "group_id": group_id,
                "text": re.sub(r"\s+", " ", claim_text).strip(),
                "label": _map_label(source_label),
                "source_label": source_label,
                "source_context": source_context,
            }
        )

    if not mapped_rows:
        raise ValueError("No usable claim_text rows found in the external dataset.")

    mapped_rows = _deduplicate_exact_text(mapped_rows)
    if not mapped_rows:
        raise ValueError(
            "No rows remain after exact-text deduplication and conflicting-label removal."
        )

    # Group near-identical templates before splitting so a rewritten variant
    # is less likely to appear in train while its sibling appears in test.
    mapped_rows = _merge_near_duplicate_groups(mapped_rows, threshold=0.70)
    output_rows = _split_by_group(mapped_rows)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["deck_id", "text", "label", "split", "notes"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    counts = {label: sum(row["label"] == label for row in output_rows) for label in LABELS}
    print(f"Wrote {len(output_rows)} external claims to {path.resolve()}")
    print("Risk-label counts: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(
        "Important: the mapped labels are a proxy derived from the source's "
        "evidentiary categories, not ground-truth startup-pitch risk labels."
    )
    read_and_validate_dataset(str(path))
    return len(output_rows), counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an external-only PitchProbe fine-tuning dataset."
    )
    parser.add_argument(
        "--output-csv",
        default="training/data/external_claims.csv",
        help="Path for the generated local CSV (ignored by Git).",
    )
    args = parser.parse_args()
    prepare_dataset(args.output_csv)


if __name__ == "__main__":
    main()
