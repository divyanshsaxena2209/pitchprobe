"""Measure the current NLI risk classifier on externally sourced test claims.

Run from the repository root:
python -m training.evaluate_baseline --data-csv training/data/external_claims.csv
"""

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.analyzer import ModelManager, classify_sentence_nli
from training.dataset_utils import LABELS, read_and_validate_dataset

BASE_MODEL = "cross-encoder/nli-deberta-v3-base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", default="training/data/external_claims.csv")
    parser.add_argument("--output-dir", default="training/artifacts/baseline")
    parser.add_argument("--model-name", default=BASE_MODEL)
    args = parser.parse_args()

    rows = read_and_validate_dataset(
        args.data_csv, require_all_training_labels=False
    )
    test_rows = [row for row in rows if row["split"] == "test"]

    manager = ModelManager.get_instance()
    manager.classifier_mode = "nli"
    manager.device = "cuda" if torch.cuda.is_available() else "cpu"
    manager.classifier_name = args.model_name
    print(f"Loading baseline classifier: {args.model_name} on {manager.device}")
    manager.classifier_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    manager.classifier_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name
    ).to(manager.device)
    manager.classifier_model.eval()
    manager.classifier_loaded = True

    predictions = []
    references = []
    output_rows = []

    for index, row in enumerate(test_rows, start=1):
        text = row["text"]
        # This evaluator measures risk assignment on dataset rows already
        # annotated as financial claims; claim/non-claim screening is evaluated
        # separately and must not be conflated with risk labels.
        risk, score = classify_sentence_nli(text, manager)
        prediction = risk.upper()
        references.append(row["label"])
        predictions.append(prediction)
        output_rows.append(
            {
                "deck_id": row["deck_id"],
                "text": text,
                "gold_label": row["label"],
                "predicted_label": prediction,
                "score": round(float(score), 4),
            }
        )
        print(f"[{index}/{len(test_rows)}] {row['label']} -> {prediction}")

    report = classification_report(
        references,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(references, predictions, labels=LABELS).tolist()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "confusion_matrix.json").write_text(
        json.dumps({"labels": LABELS, "matrix": matrix}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nBaseline results on held-out external test claims:")
    print(
        classification_report(
            references,
            predictions,
            labels=LABELS,
            target_names=LABELS,
            zero_division=0,
        )
    )
    print(f"Saved baseline report to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
