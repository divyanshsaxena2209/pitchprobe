"""Fine-tune DeBERTa for LOW / MODERATE / HIGH claim-risk labels.

Run from the repository root, preferably in a Colab GPU runtime:
python -m training.train_classifier --data-csv training/data/external_claims.csv
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from training.dataset_utils import LABELS, read_and_validate_dataset

BASE_MODEL = "cross-encoder/nli-deberta-v3-base"
SEED = 42


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, labels=list(range(len(LABELS))),
            average="macro", zero_division=0
        ),
    }


def to_dataset(rows, tokenizer, label2id):
    frame = pd.DataFrame(
        {
            "text": [row["text"] for row in rows],
            "labels": [label2id[row["label"]] for row in rows],
        }
    )
    dataset = Dataset.from_pandas(frame, preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", default="training/data/external_claims.csv")
    parser.add_argument(
        "--output-dir",
        default="training/artifacts/pitchprobe-deberta-claims",
    )
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    rows = read_and_validate_dataset(args.data_csv)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]

    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = {index: label for label, index in label2id.items()}

    print(f"Loading pretrained weights: {args.model_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(
        "The pretrained DeBERTa encoder is reused; the original NLI output "
        "head is replaced with a new three-label risk-classification head."
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.problem_type = "single_label_classification"

    train_dataset = to_dataset(train_rows, tokenizer, label2id)
    validation_dataset = to_dataset(validation_rows, tokenizer, label2id)
    test_dataset = to_dataset(test_rows, tokenizer, label2id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    predictions_output = trainer.predict(test_dataset, metric_key_prefix="test")
    predicted_ids = np.argmax(predictions_output.predictions, axis=-1)
    gold_ids = predictions_output.label_ids

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    report = classification_report(
        gold_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        gold_ids, predicted_ids, labels=list(range(len(LABELS)))
    ).tolist()

    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "confusion_matrix.json").write_text(
        json.dumps({"labels": LABELS, "matrix": matrix}, indent=2),
        encoding="utf-8",
    )

    print("\nHeld-out test results:")
    print(
        classification_report(
            gold_ids,
            predicted_ids,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            zero_division=0,
        )
    )
    print(f"Saved fine-tuned checkpoint to: {output_dir.resolve()}")
    print(
        "Do not deploy solely because training completed. Compare the held-out "
        "test report against training/artifacts/baseline first."
    )


if __name__ == "__main__":
    main()
