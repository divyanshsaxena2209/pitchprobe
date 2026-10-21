"""Fine-tune FLAN-T5 on reviewed startup-claim -> investor-question pairs.

Run from repository root:
python -m training.train_question_generator \
  --data-csv training/question_generation/claim_question_pairs.csv \
  --output-dir training/artifacts/pitchprobe-question-generator-v1

This is an experimental training script. It never reads a PDF or imports
PitchProbe analysis output. The live targeted-question rules remain unchanged.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_MODEL = "google/flan-t5-small"
LABELS = {"LOW", "MODERATE", "HIGH"}
SPLITS = {"train", "validation", "test"}
REQUIRED_COLUMNS = {
    "example_id", "claim", "risk", "dimension", "context", "question",
    "split", "source_group_id", "source_name", "source_url", "license",
    "attribution", "reviewed",
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"\s+", " ", value).strip()


def make_prompt(row: dict) -> str:
    context = str(row.get("context", "") or "").strip()
    context_line = f"\nRelevant context: {context}" if context else ""
    return (
        "Task: Write exactly one concise, specific, evidence-seeking "
        "counter-question that a skeptical seed-stage investor would ask. "
        "Probe the claim's weakest material assumption. Do not invent facts, "
        "do not repeat the claim as a question, and do not ask a generic "
        "question if a narrower probe is possible.\n"
        f"Risk category: {str(row['risk']).upper()}\n"
        f"Review dimension: {str(row['dimension']).strip()}"
        f"{context_line}\n"
        f"Claim: {str(row['claim']).strip()}\n"
        "Investor question:"
    )


def load_pairs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Training CSV not found: {path}\n"
            "Copy training/question_generation/claim_question_pairs_template.csv "
            "to claim_question_pairs.csv and fill it with human-reviewed pairs."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if df.empty:
        raise ValueError("No training examples found. Add reviewed claim-question pairs.")

    for col in REQUIRED_COLUMNS:
        if col != "context":
            df[col] = df[col].astype(str).str.strip()
    df["context"] = df["context"].astype(str).str.strip()
    df["risk"] = df["risk"].str.upper()
    df["split"] = df["split"].str.lower()

    for col in [
        "example_id", "claim", "risk", "dimension", "question", "split",
        "source_group_id", "source_name", "source_url", "license", "attribution",
    ]:
        if (df[col] == "").any():
            bad = df.index[df[col] == ""].tolist()[:10]
            raise ValueError(f"Blank required field '{col}' at CSV rows {bad}.")

    if df["example_id"].duplicated().any():
        dupes = df.loc[df["example_id"].duplicated(), "example_id"].tolist()[:10]
        raise ValueError(f"Duplicate example_id values: {dupes}")

    invalid_risks = sorted(set(df["risk"]) - LABELS)
    if invalid_risks:
        raise ValueError(f"Unsupported risk labels: {invalid_risks}")
    invalid_splits = sorted(set(df["split"]) - SPLITS)
    if invalid_splits:
        raise ValueError(f"Unsupported split names: {invalid_splits}")

    reviewed = df["reviewed"].str.lower().isin({"true", "yes", "1"})
    if not reviewed.all():
        bad = df.loc[~reviewed, "example_id"].tolist()[:10]
        raise ValueError(
            "Only human-reviewed rows may be used as targets. "
            f"Unreviewed examples: {bad}"
        )

    question_has_mark = df["question"].str.rstrip().str.endswith("?")
    if not question_has_mark.all():
        bad = df.loc[~question_has_mark, "example_id"].tolist()[:10]
        raise ValueError(f"Target questions must end with '?': {bad}")

    # Prevent rows from the same source/deck/company from crossing split boundaries.
    group_splits = df.groupby("source_group_id")["split"].nunique()
    leaked_groups = group_splits[group_splits > 1].index.tolist()
    if leaked_groups:
        raise ValueError(
            "source_group_id occurs across multiple splits (leakage): "
            f"{leaked_groups[:10]}"
        )

    # Catch exact duplicate claims crossing split boundaries, even when group IDs differ.
    df["_claim_norm"] = df["claim"].map(normalize_text)
    claim_splits = df.groupby("_claim_norm")["split"].nunique()
    duplicate_claims = claim_splits[claim_splits > 1].index.tolist()
    if duplicate_claims:
        raise ValueError(
            f"{len(duplicate_claims)} normalized claim(s) appear across splits. "
            "Deduplicate or assign the same source group and split before training."
        )

    counts = df["split"].value_counts().to_dict()
    missing_splits = SPLITS - set(counts)
    if missing_splits:
        raise ValueError(f"Missing data split(s): {sorted(missing_splits)}")
    if counts.get("train", 0) < 20 or counts.get("validation", 0) < 3 or counts.get("test", 0) < 3:
        raise ValueError(
            f"Dataset too small for even a preliminary run: split counts={counts}. "
            "Use at least 20 train, 3 validation and 3 test rows; 200–500+ reviewed "
            "pairs is a better starting target."
        )

    df = df.drop(columns=["_claim_norm"])
    print("Reviewed dataset split counts:", counts)
    print("Unique source groups:", df["source_group_id"].nunique())
    if len(df) < 200:
        print(
            "WARNING: fewer than 200 reviewed pairs. Treat any result as a smoke "
            "experiment, not evidence of generalization."
        )
    return df


def tokenized_dataset(frame: pd.DataFrame, tokenizer, max_input_length: int, max_target_length: int):
    rows = frame.to_dict(orient="records")
    data = Dataset.from_dict({
        "prompt": [make_prompt(row) for row in rows],
        "question": [row["question"] for row in rows],
    })

    def tokenize(batch):
        enc = tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=max_input_length,
        )
        targets = tokenizer(
            text_target=batch["question"],
            truncation=True,
            max_length=max_target_length,
        )
        enc["labels"] = targets["input_ids"]
        return enc

    return data.map(tokenize, batched=True, remove_columns=["prompt", "question"])


def generate_questions(model_name: str, rows: pd.DataFrame, tokenizer, device: str, max_input_length: int, max_new_tokens: int):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    prompts = [make_prompt(row) for row in rows.to_dict(orient="records")]
    generated = []
    batch_size = 4 if device == "cpu" else 8

    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(
            prompts[start:start + batch_size],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_input_length,
        ).to(device)
        with torch.inference_mode():
            output = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        generated.extend(
            tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in output
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated


def diagnostics(outputs, references):
    normalized = [normalize_text(x) for x in outputs]
    refs = [normalize_text(x) for x in references]
    return {
        "rows": len(outputs),
        "question_mark_rate": round(
            sum(str(x).rstrip().endswith("?") for x in outputs) / max(1, len(outputs)), 4
        ),
        "distinct_output_rate": round(
            len(set(normalized)) / max(1, len(normalized)), 4
        ),
        "exact_match_rate": round(
            sum(a == b for a, b in zip(normalized, refs)) / max(1, len(refs)), 4
        ),
        "mean_output_characters": round(
            sum(len(str(x)) for x in outputs) / max(1, len(outputs)), 1
        ),
        "warning": (
            "These are shallow diagnostics only. They do not measure relevance, "
            "factuality, investor usefulness, or calibration; manually review outputs."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", default="training/question_generation/claim_question_pairs.csv")
    parser.add_argument("--output-dir", default="training/artifacts/pitchprobe-question-generator-v1")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-input-length", type=int, default=384)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = load_pairs(Path(args.data_csv))
    train_rows = data[data["split"] == "train"].copy()
    validation_rows = data[data["split"] == "validation"].copy()
    test_rows = data[data["split"] == "test"].copy()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Base model: {args.base_model}")
    print(f"Device: {device}")
    print("Loading tokenizer and base seq2seq model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    train_ds = tokenized_dataset(train_rows, tokenizer, args.max_input_length, args.max_target_length)
    validation_ds = tokenized_dataset(validation_rows, tokenizer, args.max_input_length, args.max_target_length)

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=5,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    # Compare on the same untouched test claims; these are diagnostics, not
    # automatic promotion metrics. Manual blind review remains mandatory.
    del trainer, model, train_ds, validation_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Generating held-out test questions from the base model...")
    base_outputs = generate_questions(
        args.base_model, test_rows, tokenizer, device,
        args.max_input_length, args.max_new_tokens,
    )
    print("Generating held-out test questions from the fine-tuned model...")
    tuned_outputs = generate_questions(
        str(output_dir / "model"), test_rows, tokenizer, device,
        args.max_input_length, args.max_new_tokens,
    )

    side_by_side = test_rows[[
        "example_id", "claim", "risk", "dimension", "question",
        "source_group_id", "source_name", "source_url", "license", "attribution",
    ]].copy()
    side_by_side["base_model_question"] = base_outputs
    side_by_side["fine_tuned_question"] = tuned_outputs
    side_by_side.to_csv(output_dir / "heldout_question_comparison.csv", index=False)

    report = {
        "base_model": args.base_model,
        "fine_tuned_model_dir": str((output_dir / "model").resolve()),
        "split_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "base_model_diagnostics": diagnostics(base_outputs, test_rows["question"].tolist()),
        "fine_tuned_diagnostics": diagnostics(tuned_outputs, test_rows["question"].tolist()),
        "promotion_status": "NOT_PROMOTED_MANUAL_REVIEW_REQUIRED",
    }
    (output_dir / "heldout_diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nHeld-out comparison written to:", output_dir / "heldout_question_comparison.csv")
    print(json.dumps(report, indent=2))
    print(
        "\nDo not enable this checkpoint by default until the blind human review "
        "shows better claim-specific relevance and no factuality regression."
    )


if __name__ == "__main__":
    main()
