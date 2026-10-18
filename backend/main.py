import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.analyzer import ModelManager, analyze_sentences
from backend.extractor import extract_pdf_text
from backend.schemas import (
    AnalyzeResponse,
    HealthResponse,
    PipelineInfo,
    RiskSummary,
    SentenceAnalysis,
)
from backend.segmenter import split_into_sentences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pitchprobe.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize model manager and attempt to cache models
    logger.info("Initializing PitchProbe-NLP backend...")
    manager = ModelManager.get_instance()
    try:
        manager.try_load_models(background=True)
    except Exception as e:
        logger.warning("Models could not be loaded at startup: %s", e)
    yield
    # Shutdown
    logger.info("PitchProbe-NLP backend shutting down.")


app = FastAPI(
    title="PitchProbe-NLP Backend",
    description="Automated High-Risk Claim Extraction and Investor Objection Generation for Startup Pitch Decks",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify backend status and model state."""
    manager = ModelManager.get_instance()
    return HealthResponse(
        status="ok",
        classifier_loaded=manager.classifier_loaded,
        generator_loaded=manager.generator_loaded,
        device=manager.device,
        version="0.1.0",
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_pitch_deck(file: UploadFile = File(...)):
    """Analyze pitch deck PDF: extract text, segment sentences, classify risk, and generate investor objections."""
    # 1. Validate & extract PDF text
    raw_text = await extract_pdf_text(file)

    # 2. Sentence segmentation
    sentences = split_into_sentences(raw_text)
    if not sentences:
        # Fallback if text couldn't be cleanly segmented into sentences
        sentences = [s.strip() for s in raw_text.split("\n") if len(s.strip()) > 15]

    # 3. Two-stage analysis with risk gating
    results, summary_counts, pipeline_info = analyze_sentences(sentences, raw_text=raw_text)

    sentence_models = [SentenceAnalysis(**item) for item in results]
    summary_model = RiskSummary(
        total_sentences=len(sentence_models),
        low_risk=summary_counts["low_risk"],
        moderate_risk=summary_counts["moderate_risk"],
        high_risk=summary_counts["high_risk"],
    )

    return AnalyzeResponse(
        filename=file.filename or "deck.pdf",
        total_sentences=len(sentence_models),
        summary=summary_model,
        sentences=sentence_models,
        pipeline_info=PipelineInfo(**pipeline_info),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
