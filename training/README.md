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

This downloads FinClaimRadar from Hugging Face, maps its categories to `LOW`/`MODERATE`/`HIGH`, removes exact duplicate text before splitting, and drops duplicated texts whose mapped labels conflict because an identical input cannot provide a consistent single-label target. If duplicate text connects multiple source/counterfactual groups, those groups are merged before splitting so related examples cannot leak across train/validation/test. It then validates the generated data and reports class counts.

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

The training script uses the training split for updates, the validation split for macro-F1-based checkpoint selection/early stopping, and the test split for the final report. It saves the tokenizer, model, per-class precision/recall/F1, confusion matrix, and predictions.

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