# PitchProbe-NLP Commands and Architecture

## Quick Commands
- **Backend Server**: `uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload`
- **Backend Tests**: `python -m pytest backend/tests`
- **Frontend Dev**: `npm run dev`
- **Frontend Lint**: `npm run lint`
- **Frontend Build**: `npm run build`

## Architecture & Code Structure
- `backend/main.py`: FastAPI application routing `/api/health` and `POST /api/analyze`.
- `backend/extractor.py`: PDF validation (20 MB limit, format checks, in-memory stream).
- `backend/segmenter.py`: Sentence splitting with `spaCy`.
- `backend/analyzer.py`: Stage 1 3-tier NLI risk classification + Stage 2 gated question generation.
- `backend/schemas.py`: Pydantic response and request models.
- `src/components/UploadArea.tsx`: PDF uploader, loading/error states, and risk-stratified UI cards.
- `next.config.ts`: Rewrites `/api/:path*` to `http://127.0.0.1:8000/api/:path*`.

## Key Invariants
- Low-risk sentences NEVER receive counter-questions (`question: null`).
- Moderate/High risk sentences receive exactly one investor counter-question.
- No permanent disk storage of uploaded PDFs.
