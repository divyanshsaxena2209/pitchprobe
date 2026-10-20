"""Fine-tune a pairwise exaggeration classifier on public paired conclusions.

Uses only the public copenlu/scientific-exaggeration-detection dataset:
- official train split (100 pairs), split by original_file_id into train/validation
- official test split (563 pairs), used only for final evaluation

This is a separate experiment. It does not read PitchProbe uploads or its risk dataset.
Dataset card: https://huggingface.co/datasets/copenlu/scientific-exaggeration-detection
License recorded by the dataset card: GPL-3.0. Review licence obligations before
redistributing derived data or model artifacts.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "cross-encoder/nli-deberta-v3-base"
DATASET_ID = "copenlu/scientific-exaggeration-detection"
LABELS = ["same", "exaggerates", "downplays"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
SEED = 42


def select_grouped_validation_split(
    rows: Sequence[Dict[str, object]],
    validation_fraction: float = 0.20,
    seed: int = SEED,
    attempts: int = 100,
) -> Tuple[List[int], List[int]]:
    """Choose a reproducible grouped split with every label in train and validation."""
    if len(rows) < 10:
        raise ValueError("Need at least 10 training pairs for a grouped validation split.")

    labels = np.asarray([str(row["exaggeration_label"]) for row in rows])
    groups = np.asarray([str(row["original_file_id"]) for row in rows])
    if len(set(labels)) != len(LABELS):
        raise ValueError(f"Expected all labels {LABELS}; found {sorted(set(labels))}.")
    if any(not value.strip() for value in groups):
        raise ValueError("Every row must have an original_file_id for leakage-safe splitting.")

    overall = {label: float(np.mean(labels == label)) for label in LABELS}
    splitter = GroupShuffleSplit(
        n_splits=attempts,
        test_size=validation_fraction,
        random_state=seed,
    )
    best = None
    best_cost = float("inf")

    for train_idx, validation_idx in splitter.split(np.zeros(len(rows)), labels, groups):
        train_labels = set(labels[train_idx])
        validation_labels = set(labels[validation_idx])
        if train_labels != set(LABELS) or validation_labels != set(LABELS):
            continue

        validation_distribution = {
            label: float(np.mean(labels[validation_idx] == label))
            for label in LABELS
        }
        distribution_cost = sum(
            abs(validation_distribution[label] - overall[label]) for label in LABELS
        )
        size_cost = abs(len(validation_idx) / len(rows) - validation_fraction)
        cost = distribution_cost + size_cost

        if cost < best_cost:
            best = (train_idx.tolist(), validation_idx.tolist())
            best_cost = cost

    if best is None:
        raise ValueError(
            "Could not create a grouped validation split containing all three labels. "
            "Do not split by row as a workaround; inspect original_file_id groups instead."
        )

    train_groups = {groups[index] for index in best[0]}
    validation_groups = {groups[index] for index in best[1]}
    if train_groups.intersection(validation_groups):
        raise RuntimeError("Internal error: original_file_id leakage across train/validation.")

    return best


def reset_task_head(model) -> List[str]:
    """Explicitly reset same-shaped NLI head weights for the three-class target task."""
    reset_names = []
    for name in ("pooler", "classifier", "score"):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            module.apply(model._init_weights)
            reset_names.append(name)
    if "classifier" not in reset_names and "score" not in reset_names:
        raise RuntimeError("Could not find a classifier/score module to reset.")
    return reset_names


def prepare_pair_rows(dataset_rows, tokenizer, max_length: int) -> Dataset:
    """Encode press-release and abstract conclusions as an ordered text pair."""
    frame = pd.DataFrame(
        {
            "press_release_conclusion": [row["press_release_conclusion"] for row in dataset_rows],
            "abstract_conclusion": [row["abstract_conclusion"] for row in dataset_rows],
            "labels": [LABEL2ID[row["exaggeration_label"]] for row in dataset_rows],
        }
    )
    dataset = Dataset.from_pandas(frame, preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(
            batch["press_release_conclusion"],
            batch["abstract_conclusion"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["press_release_conclusion", "abstract_conclusion"],
    )


def metric_fn(eval_prediction):
    logits, gold = eval_prediction
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
    }


def report_for(gold: Sequence[int], predicted: Sequence[int]) -> Dict[str, object]:
    return classification_report(
        gold,
        predicted,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )


def run_experiment(
    output_dir: str = "training/artifacts/pitchprobe-exaggeration-pair",
    model_name: str = BASE_MODEL,
    epochs: float = 8,
    learning_rate: float = 1e-5,
    max_length: int = 384,
    seed: int = SEED,
) -> Dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Loading public dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID)
    raw_train = [dict(row) for row in dataset["train"]]
    raw_test = [dict(row) for row in dataset["test"]]
    if not raw_train or not raw_test:
        raise ValueError("Expected non-empty official train and test splits.")

    # Never allow an original study/file ID to appear across official train/test.
    train_file_ids = {str(row["original_file_id"]) for row in raw_train}
    test_file_ids = {str(row["original_file_id"]) for row in raw_test}
    official_overlap = train_file_ids.intersection(test_file_ids)
    if official_overlap:
        raise ValueError(
            f"Dataset official train/test splits share {len(official_overlap)} "
            "original_file_id values. Stop and inspect those rows before evaluation."
        )

    train_indices, validation_indices = select_grouped_validation_split(raw_train, seed=seed)
    train_rows = [raw_train[index] for index in train_indices]
    validation_rows = [raw_train[index] for index in validation_indices]
    train_groups = {str(row["original_file_id"]) for row in train_rows}
    validation_groups = {str(row["original_file_id"]) for row in validation_rows}
    print(
        f"Official training pool: {len(raw_train)}; internal train: {len(train_rows)}; "
        f"validation: {len(validation_rows)}; official held-out test: {len(raw_test)}"
    )
    print("Internal source-group leakage:", bool(train_groups.intersection(validation_groups)))
    print("Training labels:", dict(Counter(row["exaggeration_label"] for row in train_rows)))
    print("Validation labels:", dict(Counter(row["exaggeration_label"] for row in validation_rows)))
    print("Official test labels:", dict(Counter(row["exaggeration_label"] for row in raw_test)))
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Majority-class baseline is determined solely from the official training pool.
    majority_label = Counter(row["exaggeration_label"] for row in raw_train).most_common(1)[0][0]
    test_gold = [LABEL2ID[row["exaggeration_label"]] for row in raw_test]
    majority_predictions = [LABEL2ID[majority_label]] * len(raw_test)
    majority_report = report_for(test_gold, majority_predictions)
    print(f"Majority-class baseline (predict {majority_label} for all):")
    print(
        f"accuracy={accuracy_score(test_gold, majority_predictions):.4f}, "
        f"macro_f1={f1_score(test_gold, majority_predictions, labels=list(range(3)), average='macro', zero_division=0):.4f}"
    )

    print(f"Loading paired-text encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    reset_modules = reset_task_head(model)
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.config.problem_type = "single_label_classification"
    print("Reinitialized task head:", ", ".join(reset_modules))

    train_ds = prepare_pair_rows(train_rows, tokenizer, max_length)
    validation_ds = prepare_pair_rows(validation_rows, tokenizer, max_length)
    test_ds = prepare_pair_rows(raw_test, tokenizer, max_length)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output),
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=5,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metric_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    # Avoid the eval_macro_f1 early-stopping warning during test_ metric evaluation.
    trainer.remove_callback(EarlyStoppingCallback)
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    prediction_output = trainer.predict(test_ds, metric_key_prefix="test")
    predicted_ids = np.argmax(prediction_output.predictions, axis=-1)

    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))

    majority_matrix = confusion_matrix(test_gold, majority_predictions, labels=list(range(3))).tolist()
    model_matrix = confusion_matrix(test_gold, predicted_ids, labels=list(range(3))).tolist()
    model_report = report_for(test_gold, predicted_ids)
    pd.DataFrame(
        {
            "original_file_id": [row["original_file_id"] for row in raw_test],
            "press_release_conclusion": [row["press_release_conclusion"] for row in raw_test],
            "abstract_conclusion": [row["abstract_conclusion"] for row in raw_test],
            "gold_label": [row["exaggeration_label"] for row in raw_test],
            "predicted_label": [ID2LABEL[int(index)] for index in predicted_ids],
        }
    ).to_csv(output / "test_predictions.csv", index=False)

    results = {
        "dataset": DATASET_ID,
        "dataset_license": "GPL-3.0 as listed on the Hugging Face dataset card; review before redistribution",
        "base_model": model_name,
        "seed": seed,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "epochs_requested": epochs,
        "official_train_rows": len(raw_train),
        "internal_train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "official_test_rows": len(raw_test),
        "majority_baseline_label": majority_label,
        "majority_baseline_accuracy": float(accuracy_score(test_gold, majority_predictions)),
        "majority_baseline_macro_f1": float(f1_score(test_gold, majority_predictions, labels=list(range(3)), average="macro", zero_division=0)),
        "model_accuracy": float(accuracy_score(test_gold, predicted_ids)),
        "model_macro_f1": float(f1_score(test_gold, predicted_ids, labels=list(range(3)), average="macro", zero_division=0)),
    }
    (output / "experiment_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "classification_report.json").write_text(json.dumps(model_report, indent=2), encoding="utf-8")
    (output / "majority_baseline_report.json").write_text(json.dumps(majority_report, indent=2), encoding="utf-8")
    (output / "confusion_matrix.json").write_text(
        json.dumps({"labels": LABELS, "matrix": model_matrix}, indent=2), encoding="utf-8"
    )
    (output / "majority_confusion_matrix.json").write_text(
        json.dumps({"labels": LABELS, "matrix": majority_matrix}, indent=2), encoding="utf-8"
    )
    print("\nPAIRWISE EXAGGERATION TEST RESULTS")
    print(classification_report(test_gold, predicted_ids, labels=list(range(3)), target_names=LABELS, zero_division=0))
    print(
        f"Fine-tuned accuracy={results['model_accuracy']:.4f}; "
        f"macro-F1={results['model_macro_f1']:.4f}"
    )
    print(
        f"Majority baseline accuracy={results['majority_baseline_accuracy']:.4f}; "
        f"macro-F1={results['majority_baseline_macro_f1']:.4f}"
    )
    print("Saved model and reports to:", output.resolve())
    print("Research result only: do not merge into PitchProbe master without independent validation.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a paired-text exaggeration classifier.")
    parser.add_argument("--output-dir", default="training/artifacts/pitchprobe-exaggeration-pair")
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_experiment(
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
