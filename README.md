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

Configure these backend environment variables only after exporting and checking a compatible fine-tuned risk checkpoint:

- `PITCHPROBE_CLASSIFIER_PATH`: local checkpoint directory or Hub model ID (for example, `your-hf-user/pitchprobe-deberta-risk`).
- `PITCHPROBE_CLASSIFIER_REVISION`: recommended Hub commit SHA to pin the exact model revision.
- `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`): read token if the model repository is private. Keep it in deployment secrets; never commit it.

The loader accepts **exactly three output labels**: `LOW`, `MODERATE`, and `HIGH`, with output IDs `0`, `1`, and `2`. A checkpoint from the scientific exaggeration pair experiment (`same`, `exaggerates`, `downplays`) is incompatible with this single-claim risk classifier and will be rejected; the backend then falls back to the original NLI classifier.

The FinClaimRadar-derived model uses mapped evidentiary categories as proxy labels, not independently validated startup-pitch risk labels. Before enabling it for users, inspect its held-out metrics and evaluate on a separate, human-reviewed pitch-claim set. A successful load means the model files and label schema are compatible; it does **not** prove the model is more accurate. Review the source dataset's CC BY-SA 4.0 terms before redistributing model weights.

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
