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
