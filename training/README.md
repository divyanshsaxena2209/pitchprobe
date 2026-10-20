# PitchProbe fine-tuning using public external datasets

This workflow uses public external datasets only. It never reads `sonora_seed_pitch_deck.pdf`, PitchProbe output, or any files you upload locally. The locally generated CSV and trained checkpoint are ignored by Git.

## What is being fine-tuned

Base checkpoint: `cross-encoder/nli-deberta-v3-base`.

The supervised risk head has three labels:

- `LOW`: claims presented as specific, sourced historical facts.
- `MODERATE`: broad market observations, attributed analysis, or forward-looking statements that require qualification or validation.
- `HIGH`: speculative rumours and unsupported/hype-oriented assertions.

Claim/non-claim screening remains a separate step in `backend/claim_detector.py`. This is deliberate: the external financial-claim dataset labels types of evidence, not whether each line is a startup claim. We do not invent a `NON_CLAIM` class label for a dataset that contains only claims.

**Important limitation:** these source categories are only proxies for PitchProbe's risk labels. The mapping is useful to establish a reproducible experiment, but it is not ground-truth startup-pitch risk annotation. The resulting model must not be described as validated for pitch decks until it has been evaluated on a separate, appropriately licensed external startup/pitch dataset with risk labels.

## Primary dataset: FinClaimRadar

Source: [FinClaimRadar on Hugging Face](https://huggingface.co/datasets/ImBinarry/FinClaimRadar-Financial-Reasoning).

The dataset card lists 630 English financial claims with six evidentiary categories and a `CC-BY-SA-4.0` licence. The claims were model-generated initially and then refined/reviewed by the dataset author. It is a small proof-of-concept, not a large or definitive pitch-deck benchmark.

The preparation script maps the six source categories as follows:

| FinClaimRadar source category | PitchProbe training label | Rationale |
|---|---|---|
| Reported Factual Data | `LOW` | Specific past result tied to a stated, verifiable source |
| General Market Observation/Trend Description | `MODERATE` | Broad/non-specific trend statements may need a precise basis |
| Attributed Expert Opinion/Analysis | `MODERATE` | Interpretation or forecast rather than direct historical measurement |
| Company Forward-Looking Statement | `MODERATE` | Targets/projections need assumptions and evidence |
| Speculative Statement/Rumor | `HIGH` | Unconfirmed information |
| Unsupported Assertion/Hype | `HIGH` | Unsupported or exaggerated claim |

This mapping is intentionally explicit and imperfect. Never interpret it as the original dataset's official label mapping. The generated CSV records the original category in `notes` so mistakes can be audited.

## Run the external-only training workflow

Use Google Colab with a GPU if available; a GPU is recommended, not guaranteed. Start from the repository root in the Colab runtime, or upload the repository ZIP to the runtime.

### 1. Install dependencies

```bash
pip install -r training/requirements.txt
```

### 2. Download and split the public dataset

```bash
python -m training.prepare_finclaimradar --output-csv training/data/external_claims.csv
```

This downloads FinClaimRadar from Hugging Face, maps its categories to `LOW`/`MODERATE`/`HIGH`, removes exact duplicate text before splitting, and drops duplicated texts whose mapped labels conflict because an identical input cannot provide a consistent single-label target. If duplicate text connects multiple source/counterfactual groups, those groups are merged. It also merges rows into the same split group when character TF-IDF cosine similarity is at least 0.65. This is a conservative near-template grouping heuristic, not proof that claims have identical meaning; it does not delete these near-duplicate rows. The validator then checks for exact cross-split duplicates and group leakage.

### 3. Measure the original classifier first

```bash
python -m training.evaluate_baseline --data-csv training/data/external_claims.csv
```

The evaluator runs the original NLI risk classifier directly on the held-out external claims. It does not apply the separate claim/non-claim screen to these rows; otherwise the baseline would mix two tasks and treat a skipped claim as a risk label.

### 4. Fine-tune DeBERTa

```bash
python -m training.train_classifier --data-csv training/data/external_claims.csv
```

Default checkpoint output: `training/artifacts/pitchprobe-deberta-claims/`.

The training script uses the training split for updates, the validation split for macro-F1-based checkpoint selection/early stopping, and the test split for the final report. It explicitly reinitializes the DeBERTa task-specific pooler/classifier modules: the source NLI model already has three output classes, so `ignore_mismatched_sizes=True` alone would not guarantee a fresh risk head. The encoder weights remain pretrained. It saves the tokenizer, model, per-class precision/recall/F1, confusion matrix, and predictions.

### 5. Decide whether the experiment helped

Compare:

- `training/artifacts/baseline/classification_report.json`
- `training/artifacts/pitchprobe-deberta-claims/classification_report.json`

Both reports must be based on the same held-out external examples. Inspect per-class precision/recall and misclassifications. Because the labels are mapped proxies and the dataset is small, good numbers on this split are **not** proof that unseen startup pitch decks will be analysed accurately. Do not deploy a worse or unvalidated checkpoint just because training completed.

## Other public datasets and what they can (and cannot) do

- [ClaimBuster](https://huggingface.co/datasets/Nithiwat/claimbuster) contains 23,533 statements from US presidential debates labelled for check-worthiness. It may be useful for a separate claim-screening experiment, but its political-debate domain is different from startup decks; do not map its check-worthiness labels directly to `LOW`/`MODERATE`/`HIGH`.
- [FinQA](https://huggingface.co/datasets/ibm-research/finqa) contains around 8,000 financial question-answer pairs based on financial reports and is listed as `CC-BY-4.0`. It can help with finance-specific question/answer behaviour, but its questions are numerical reasoning questions, not venture-capital objections.
- [TAT-QA](https://github.com/NExTplusplus/TAT-QA) contains 16,552 questions over financial-report text and tables. It is another financial QA source, not a dataset of investor counter-questions.
- [FEVER](https://huggingface.co/datasets/EleutherAI/fever) contains claims labelled Supported, Refuted, or NotEnoughInfo. It supports evidence-verification experiments, but Wikipedia claim veracity is not the same task as judging a pitch's vagueness or hype.

Before using any additional source, check its own dataset card and licence, cite the source, and do not combine labels that mean different things.

## Question-generator fine-tuning (separate phase)

The base `google/flan-t5-small` produced generic questions in the observed result, so the app now uses targeted question rules by default. FLAN-T5 is opt-in with `PITCHPROBE_USE_FLAN_T5=1` for experiments.

Useful public sources for a **future external-only question-generation dataset**:

- [Startup Investor Falsification Question Bank](https://huggingface.co/datasets/mheilimo/startup-investor-falsification-question-bank): 48 original evidence-seeking questions across 12 private-company review dimensions; the dataset card lists **CC BY 4.0** and requires attribution to Playful Pixels Oy / DDScore.ai. It contains question templates and evidence requests, but not a matched claim-plus-target-question pair for every example. Use it as an externally sourced question bank, not as a ready-made seq2seq training set.
- [Private-company first-pass due-diligence checklist](https://huggingface.co/datasets/mheilimo/private-company-due-diligence-checklist): 12 dimension-level review prompts; the card lists **CC BY 4.0**. It is a useful source of topic coverage, but 12 rows are not enough to fine-tune a generator.
- [FinQA](https://huggingface.co/datasets/ibm-research/finqa): around 8,000 questions/answers from financial reports, listed as **CC BY 4.0**. It can provide financial question style and numerical reasoning examples, but these are not venture-capital challenge questions.

**Do not fine-tune FLAN-T5 on those questions alone.** Seq2seq training needs source-target pairs. A useful pair would contain the claim, the risk category, relevant evidence/context, and a reviewed investor question as the target. The public sources above do not provide enough directly matched, high-quality pairs for PitchProbe's exact task. The honest next step is to curate an external-only paired dataset from public claims and the question bank, review the claim-question matches, and keep entire source groups separate between train/validation/test. Until then, keep the current targeted-question fallback rather than training on unrelated questions or self-generated PitchProbe outputs.

## Load a fine-tuned classifier locally (only after evaluation)

After training and reviewing the held-out report, copy the checkpoint folder to your local PitchProbe project. In the backend PowerShell terminal:

```powershell
$env:PITCHPROBE_CLASSIFIER_PATH = "C:\Users\divya\OneDrive\Desktop\pitchprobe\training\artifacts\pitchprobe-deberta-claims"
python .\start_backend.py
```

The backend expects a three-label classifier config containing `LOW`, `MODERATE`, and `HIGH`. It keeps the heuristic claim screen separate and uses the fine-tuned model only for risk assignment. If the checkpoint fails to load, the backend logs the problem and falls back to the original NLI model.

To use the original classifier again:

```powershell
Remove-Item Env:PITCHPROBE_CLASSIFIER_PATH -ErrorAction SilentlyContinue
python .\start_backend.py
```

Remove or change the environment variable in the same terminal before restarting the backend.


## Paired exaggeration-detection experiment (separate task)

This experiment is deliberately separate from the FinClaimRadar risk classifier. It uses the public [Scientific Exaggeration Detection dataset](https://huggingface.co/datasets/copenlu/scientific-exaggeration-detection), listed as GPL-3.0 by its dataset card. Review GPL-3.0 requirements before redistributing derived data or model artifacts. The dataset contains paired \`press_release_conclusion\` and \`abstract_conclusion\` text with \`same\`, \`exaggerates\`, or \`downplays\` labels. It is a scientific/health communication benchmark, not a startup-pitch dataset.

The script uses the official 100-row train split, creates a grouped internal train/validation split by \`original_file_id\`, and reserves the official 563-row test split for final evaluation only. It refuses to continue if source IDs overlap between official train and test. It resets the task-specific three-class head while reusing the pretrained DeBERTa encoder. It reports a majority-class baseline, per-class metrics, confusion matrix, and test predictions.

From Colab's repository root:

\`\`\`bash
python -m pytest training/test_exaggeration_pair.py -q
python -m training.train_exaggeration_pair --output-dir training/artifacts/pitchprobe-exaggeration-pair
\`\`\`

The run should take a few minutes on a T4 GPU, subject to Colab availability. Outputs go to \`training/artifacts/pitchprobe-exaggeration-pair/\` (model/tokenizer, \`experiment_metrics.json\`, \`classification_report.json\`, \`confusion_matrix.json\`, \`majority_baseline_report.json\`, and \`test_predictions.csv\`). They are ignored by Git. Do not use the official test split for checkpoint selection, and do not merge this model into the application. Its scores measure scientific press-release exaggeration, not pitch-deck claim risk.

## Published scientific claim-strength baseline (Colab results, 2026-10-21)

A second, inference-only benchmark used the published model [`copenlu/scientific-exaggeration-detection`](https://huggingface.co/copenlu/scientific-exaggeration-detection). The model card describes it as a claim-strength predictor intended to compare the strength of two scientific sentences; the underlying paper evaluates this strategy for scientific press-release exaggeration ([Wright & Augenstein, EMNLP 2021](https://arxiv.org/abs/2108.13493)).

Method: independently classify the `press_release_conclusion` and `abstract_conclusion` into the four ordered strength levels (0 = no relationship, 1 = correlation, 2 = conditional causation, 3 = causation), then compare predicted levels: release strength above abstract = `exaggerates`, below = `downplays`, equal = `same`. The official 563-row test split was checked for row alignment against the saved test predictions before evaluation. This run performed inference only and did not change the trained DeBERTa pairwise checkpoint.

### Observed metrics

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority baseline (always `same`) | 0.6128 | 0.2533 |
| Fine-tuned DeBERTa pair classifier | 0.5542 | 0.2875 |
| Published strength classifier + strength-gap rule | **0.6785** | **0.6217** |

Published strength-classifier results by paired label:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `same` | 0.795 | 0.730 | 0.761 | 345 |
| `exaggerates` | 0.570 | 0.664 | 0.614 | 122 |
| `downplays` | 0.471 | 0.510 | 0.490 | 96 |

Confusion matrix (rows = actual; columns = predicted; order `same`, `exaggerates`, `downplays`):

```text
              same  exaggerates  downplays
same           252           49         44
exaggerates     30           81         11
downplays       35           12         49
```

Individual strength accuracy reported by the evaluation cell: press-release strength 0.6785, abstract strength 0.7353, combined 0.7069. Artifacts were saved locally in Colab as `published_strength_model_predictions.csv` and `published_strength_model_metrics.json` under `training/artifacts/pitchprobe-exaggeration-pair/`; generated outputs are ignored by Git.

### Interpretation and caveats

- This is the strongest result so far for this **scientific exaggeration benchmark**, improving on both the majority baseline and the fine-tuned pair classifier.
- The model is an existing published model built for causal-claim-strength comparison and related scientific exaggeration detection. The authors state that this benchmark's test file was used in the experiments reported in their paper, so treat these numbers as an in-domain reproduction/reference baseline—not an independent blind test of the published model, and not evidence of broad out-of-domain generalization.
- This task concerns scientific/health press-release conclusions versus abstracts, not startup pitch decks or investor claim risk. It must not be integrated into the PitchProbe application without separate task-specific validation.
- The dataset card lists GPL-3.0 for the dataset. Review applicable dataset and model licenses before redistributing data or model artifacts.

## Startup-risk and investor-question next steps (research-only)

### Current dataset decision

A current source scan did **not** identify a sufficiently sized, human-labelled startup-pitch dataset with PitchProbe's exact LOW/MODERATE/HIGH risk definitions. Do not treat FinClaimRadar's evidentiary-basis labels as gold startup risk labels. The [Startup Investor Falsification Question Bank](https://huggingface.co/datasets/mheilimo/startup-investor-falsification-question-bank) is more directly relevant to question quality: it provides 48 review prompts across 12 dimensions, with an evidence request, contradiction signal and stage note. Its card lists **CC BY 4.0**; attribute Playful Pixels Oy and link the license when redistributing/adapting the source.

### Question-bank retrieval prototype

training/question_bank_router.py provides a training-only retrieval baseline; it does not use a generative model or fine-tune on the 48 questions. It routes common investor claim topics to a review dimension, then ranks source questions using TF-IDF over dimension, claim type, question, evidence request and contradiction signal. It returns the original prompt plus supporting evidence fields. A retrieval match is a suggestion for reviewer approval, not verification or truth scoring.

Run the unit checks:

    python -m pytest training/test_question_bank_router.py -q

To retrieve prompts for a CSV with a claim column:

    python -m training.question_bank_router --claims-csv training/data/example_claims.csv --output-csv training/artifacts/question_bank_matches.csv --top-k 3

The default source URL is pinned to the dataset's published revision f39e258288f16c6f85dc40e67d38b9e90eff1426. The prototype needs human evaluation for claim-specific relevance and question usefulness before application integration.

### Create a human-reviewed risk-label sheet

training/create_risk_review_sheet.py samples only rows in the CSV's training split and adds blank annotation fields. It intentionally keeps the original proxy label visible for audit, but the annotator must assign labels independently using the rubric. A second review/adjudication column is provided. NEEDS_CONTEXT means not enough context to label fairly and should not silently be converted into a risk class.

Run its checks and generate the worksheet:

    python -m pytest training/test_create_risk_review_sheet.py -q
    python -m training.create_risk_review_sheet --input-csv training/data/external_claims_neardedup.csv --output-csv training/data/manual_risk_review.csv --sample-size 60

This creates an annotation worksheet, **not** a validated benchmark. FinClaimRadar is still a financial-claim proxy dataset rather than real startup pitch-deck claims, and its card lists CC BY-SA 4.0. The next credible classifier experiment should use independently reviewed labels, preserve source-group splits, report inter-annotator agreement where feasible, and retain a fully untouched test set. Do not fine-tune or integrate based on the original mapped proxy labels alone.

