# PitchProbe classifier fine-tuning (Phase 1)

This folder contains a **baseline evaluation** and a supervised fine-tuning pipeline for the existing DeBERTa checkpoint. It does not automatically generate labels or train on the model's own predictions.

## What we are fine-tuning

Base checkpoint: `cross-encoder/nli-deberta-v3-base`.

The training script reuses the pretrained DeBERTa encoder and replaces the original 3-way NLI head with a 4-way classifier:

- `NON_CLAIM`: headings, references, administrative text, and background that is not an investor-facing assertion.
- `LOW`: an investor-facing assertion with specific supporting evidence shown in the deck. This does **not** mean independently verified as true.
- `MODERATE`: a relevant claim that is vague, underspecified, or missing important validation.
- `HIGH`: a highly exaggerated, absolute, implausible, or strongly unsupported claim.

These labels are judgment-based. Create written annotation rules and apply them consistently. Don't label every numerical statement as LOW: unsupported projections can still be HIGH. Mark uncertainty for human review instead of guessing.

## Step 1 — Build the labeled dataset

Copy `training/data/claims_template.csv` to `training/data/claims.csv` and add reviewed rows. Required columns:

| Column | Purpose |
|---|---|
| `deck_id` | Stable ID for the source deck, e.g. `sonora_01`; do not use a company name if confidentiality matters. |
| `text` | One claim-sized sentence or short slide bullet. Preserve relevant numbers and qualifiers. |
| `label` | Exactly one of `NON_CLAIM`, `LOW`, `MODERATE`, `HIGH`. |
| `split` | `train`, `validation`, or `test`. Every row from the same deck must stay in the same split. |
| `notes` | Optional rationale or evidence cited from the deck. |

For an initial pilot, aim for roughly 500–1,000 reviewed examples across multiple decks, reasonably balanced across all four labels. This is a starting target, not a guarantee of good results. Include difficult examples from different industries and deck styles. Avoid copying near-identical sentences into different splits.

**Keep decks separated by split.** For example, use some complete decks for training, different complete decks for validation, and different unseen decks for testing. The validation/test split must not contain slides or paraphrases from training decks.

Do not commit private pitch decks, confidential claim text, or `claims.csv` to GitHub. The raw deck should stay on your own device unless you have permission to share it. The shared repository contains only the empty template.

## Step 2 — Install training dependencies

Use a GPU runtime such as Google Colab when available. Colab GPU access and session limits vary; a GPU is recommended, not guaranteed.

From a fresh Colab notebook, mount Google Drive or upload the repository and dataset. Then install the listed dependencies in a dedicated runtime:

```bash
pip install -r training/requirements.txt
```

Check the GPU:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

If you are running from the repository root, these commands validate the data and measure the current model before training:

```bash
python -m training.dataset_utils
python -m training.evaluate_baseline --data-csv training/data/claims.csv
```

The baseline script loads the current NLI checkpoint, applies the current claim screen and risk classifier, and writes per-class precision/recall/F1, confusion matrix, and predictions under `training/artifacts/baseline/`.

## Step 3 — Fine-tune DeBERTa

```bash
python -m training.train_classifier --data-csv training/data/claims.csv
```

Default output: `training/artifacts/pitchprobe-deberta-claims/`.

The trainer:
- Uses the training split for gradient updates only.
- Uses validation macro-F1 to select/checkpoint the model and early-stop.
- Evaluates the final best checkpoint once against the separate test split.
- Saves model/tokenizer and test metrics, per-class report, and confusion matrix.

Check macro-F1 and the per-class report, especially recall for HIGH and precision for NON_CLAIM. Don't deploy if the model improves aggregate F1 but becomes substantially worse at detecting high-risk claims.

## Step 4 — Compare before deployment

Compare:
- `training/artifacts/baseline/classification_report.json`
- `training/artifacts/pitchprobe-deberta-claims/classification_report.json`

These comparisons are meaningful only if both evaluate the exact same held-out test rows. Review misclassified examples manually. Keep the original NLI model as a fallback until the fine-tuned checkpoint shows a useful improvement on unseen decks.

## Phase 2 — Not included yet

Fine-tuning FLAN-T5 for investor-question generation should happen only after the classifier baseline and annotation set are reliable. It needs a separate curated dataset of claim + relevant context + risk label + human-approved investor question. Do not train on unreviewed generic questions produced by the current model.


## Step 5 — Load the fine-tuned checkpoint locally (after evaluation)

Only do this after you review the held-out test report and decide the fine-tuned classifier is better than the baseline.

In the PowerShell terminal used to launch the backend, set the local checkpoint path before starting the launcher:

```powershell
$env:PITCHPROBE_CLASSIFIER_PATH = "C:\Users\divya\OneDrive\Desktop\pitchprobe\training\artifacts\pitchprobe-deberta-claims"
python .\start_backend.py
```

The backend will load this checkpoint only if its config contains all four expected labels. If loading fails, it logs the issue and falls back to the original NLI model. The environment variable must be present in the terminal that starts the backend. It is not committed to Git.

To return to the original classifier in the same PowerShell session:

```powershell
Remove-Item Env:PITCHPROBE_CLASSIFIER_PATH -ErrorAction SilentlyContinue
python .\start_backend.py
```
