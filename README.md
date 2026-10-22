# PitchProbe-NLP (MVP)

Automated High-Risk Claim Extraction and Investor Objection Generation for Startup Pitch Decks, built according to the DA1 research specification.

## Architecture Overview
- **Frontend**: Next.js 16 (App Router), TypeScript, Tailwind CSS.
- **Backend**: Python FastAPI with `pdfplumber` (text extraction) and `spaCy` (sentence splitting).
- **Stage 0 (Claim screening)**: Filters headings, references, administrative text, and ordinary background before scoring. Clear business claims use domain signals; ambiguous sentences can use the already-loaded NLI model. Likely academic assignments/research reports return a warning instead of misleading investor scores.
- **Stage 1 (Risk classification)**: Only detected investor-facing claims receive a 3-tier risk label (Low / Grounded, Moderate / Vague, High / Hyperbolic) using `cross-encoder/nli-deberta-v3-base` and calibrated domain fallback.
- **Stage 2 (Question generation)**: Gated investor objection counter-questions with `google/flan-t5-small`. Low-risk claims receive no questions; Moderate/High claims receive targeted counter-questions.

---

## 1. Backend Setup & Run

### Install Dependencies
```bash
python -m pip install fastapi uvicorn python-multipart pdfplumber spacy transformers "protobuf<5.0.0" reportlab pytest httpx
```

### Run Backend Server
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Health Check: `GET http://127.0.0.1:8000/api/health`
- Analysis Endpoint: `POST http://127.0.0.1:8000/api/analyze`

### Run Backend Tests
```bash
python -m pytest backend/tests
```

---

## 2. Frontend Setup & Run

### Run Next.js Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build & Lint Checks
```bash
npm run lint
npm run build
```

---

## 3. Usage
1. Open [http://localhost:3000](http://localhost:3000).
2. Choose or drag-and-drop a pitch deck PDF (up to 20 MB, e.g. `public/pitch_deck_example.pdf`).
3. Click **Analyze pitch deck**.
4. View only detected investor-facing claims, their risk labels, confidence scores, and targeted counter-questions. If the PDF looks like coursework or a research report, PitchProbe explains that it skipped investor scoring.


## 4. Model Fine-Tuning (Phase 1)

PitchProbe includes a four-label DeBERTa fine-tuning pipeline and a baseline evaluator in [training/README.md](training/README.md). The first step is to create a human-reviewed, deck-separated dataset; do not train on unreviewed predictions from the current model. Private dataset files and trained checkpoints are ignored by Git.


## 5. Optional fine-tuned DeBERTa risk classifier

The backend can load a three-label risk classifier from a local checkpoint or a Hugging Face Hub model repository. This is **opt-in**; with no `PITCHPROBE_CLASSIFIER_PATH`, PitchProbe continues using the original `cross-encoder/nli-deberta-v3-base` classification path. The question-generation code and its default targeted-rule behavior are unchanged.

### Local Windows staging launcher

Use a disposable staging branch first; this does not turn on the fine-tuned classifier in your current production branch by itself.

1. In PowerShell, open the PitchProbe repository, check that you have no uncommitted changes, then fetch and switch to `staging/local-deberta-smoke-test`.
2. Install the backend dependencies if needed: `python -m pip install -r backend/requirements.txt`.
3. Start the staging backend with `python .\start_staging_backend.py`.
4. When prompted, enter a dedicated Hugging Face read-only token with access to the private checkpoint. The script validates the pinned revision and LOW/MODERATE/HIGH label mapping, starts the regular Windows-aware backend launcher, waits for model loading, then checks `/api/health` reports the exact model and `classifier_mode=fine_tuned`.
5. If the backend falls back to NLI or the checkpoint fails to load, the staging launcher stops instead of silently letting you assume the fine-tuned model is active. On successful verification, keep the terminal open; `Ctrl+C` stops staging.

The script never writes the token to `.env.local` or the repository. Question generation is kept on the existing targeted-question rules through `PITCHPROBE_USE_FLAN_T5=0`. Do not merge this staging-only branch to `master` until the local smoke test passes.

### Staging checkpoint selected (2026-10-21)

The current experimental checkpoint is hosted privately at `divstu/pitchprobe-deberta-claims-verified-v2`, pinned to revision `5771ca4cdbc6afc05bdb0baec92409c7f02e9be6`. The exact non-secret settings are in [`backend/.env.staging.example`](backend/.env.staging.example). Do not commit an access token.

To stage it, configure these environment variables on the **backend runtime only**:
- `PITCHPROBE_CLASSIFIER_PATH=divstu/pitchprobe-deberta-claims-verified-v2`
- `PITCHPROBE_CLASSIFIER_REVISION=5771ca4cdbc6afc05bdb0baec92409c7f02e9be6`
- `HF_TOKEN`: a dedicated Hugging Face **read-only** token stored as a hosting-provider secret (do not reuse the write token for runtime access).
- `PITCHPROBE_USE_FLAN_T5=0` so question generation remains on targeted rules.

After setting them, redeploy/restart the backend and verify startup logs say the fine-tuned classifier loaded; if loading fails, the existing code falls back to NLI. Test several known sample claims and confirm the reported pipeline is `fine_tuned_three_label+targeted_question_rules` before using this outside staging.

Configure these backend environment variables only after exporting and checking a compatible fine-tuned risk checkpoint:

- `PITCHPROBE_CLASSIFIER_PATH`: local checkpoint directory or Hub model ID (for example, `your-hf-user/pitchprobe-deberta-risk`).
- `PITCHPROBE_CLASSIFIER_REVISION`: recommended Hub commit SHA to pin the exact model revision.
- `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`): read token if the model repository is private. Keep it in deployment secrets; never commit it.

The loader accepts **exactly three output labels**: `LOW`, `MODERATE`, and `HIGH`, with output IDs `0`, `1`, and `2`. A checkpoint from the scientific exaggeration pair experiment (`same`, `exaggerates`, `downplays`) is incompatible with this single-claim risk classifier and will be rejected; the backend then falls back to the original NLI classifier.

The FinClaimRadar-derived model uses mapped evidentiary categories as proxy labels, not independently validated startup-pitch risk labels. Before enabling it for users, inspect its held-out metrics and evaluate on a separate, human-reviewed pitch-claim set. A successful load means the model files and label schema are compatible; it does **not** prove the model is more accurate. Review the source dataset's CC BY-SA 4.0 terms before redistributing model weights.

**Latest proxy-label benchmark (2026-10-21; research checkpoint, not yet the default):** On the current 64-example held-out split of the prepared FinClaimRadar-derived CSV (SHA-256 `a95da8cf9622850fb112d73863728e85fcef5c49e8be5289d93cf419cb71ac26`), the fresh fine-tuned checkpoint scored 0.984 accuracy and 0.983 macro-F1; the original NLI baseline scored 0.312 accuracy and 0.227 macro-F1 on the same examples. Fine-tuned class F1 was LOW=0.966, MODERATE=0.984, HIGH=1.000. This is encouraging **only for this small, mapped-proxy test set** (14 LOW, 32 MODERATE, 18 HIGH); it does not establish performance on independently labelled startup-pitch claims. The checkpoint and report are stored in Google Drive/Colab, not committed to Git. Enable only in a controlled staging test after publishing the checkpoint to an access-controlled model repository, pinning its revision, verifying runtime loading, and reviewing the dataset/model licence implications.

To keep question generation unchanged, leave `PITCHPROBE_USE_FLAN_T5=0` (the default). Fine-tuned risk classification only replaces the risk-label prediction when `PITCHPROBE_CLASSIFIER_PATH` is set and the checkpoint loads successfully.

### Long-running PDF analysis

The browser uses `NEXT_PUBLIC_BACKEND_URL` to post PDFs directly to FastAPI, bypassing the Next.js rewrite proxy for long CPU-bound analysis requests. `start_backend.py` writes both `BACKEND_URL` and `NEXT_PUBLIC_BACKEND_URL` whenever it confirms the selected port. Always start/restart Next.js after starting the launcher or if the backend selects a new port, because public environment variables are included when the client bundle is built.

### Windows stability / CPU troubleshooting

The backend launcher advances to the next port after every failed candidate. It marks a port successful only after the child Uvicorn process itself logs that it is listening and that port's health endpoint responds; it does not mistake an older process on the same port for the new child. If a previously healthy backend later crashes, the launcher also advances to a new port and updates `BACKEND_URL`. Restart the Next.js dev server after that runtime port change so its rewrite configuration reloads the updated URL.

If Python exits unexpectedly during model inference on Windows, test with CPU inference to isolate the CUDA/GPU runtime:

```powershell
$env:PITCHPROBE_DEVICE = "cpu"
python .\start_backend.py
```

This affects the backend started from that PowerShell session and keeps the pretrained model weights unchanged; CPU inference may be slower. If CPU mode is stable, investigate the installed PyTorch/CUDA/driver stack before returning to automatic GPU selection. Remove the override in that terminal with `Remove-Item Env:PITCHPROBE_DEVICE` to return to automatic device selection.
