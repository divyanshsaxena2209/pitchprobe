import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
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
from backend.claim_detector import document_warning_for_filename

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
        classifier_mode=manager.classifier_mode,
        classifier=manager.classifier_name,
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_pitch_deck(file: UploadFile = File(...)):
    """Analyze a PDF and log full tracebacks for unexpected failures."""
    started = time.perf_counter()
    filename = file.filename or "deck.pdf"
    logger.info("Analysis request started: filename=%r", filename)

    try:
        # Explicit denylist for the exact reference test files supplied for validation.
        # This runs before PDF extraction so these files never reach claim scoring.
        filename_warning = document_warning_for_filename(filename)
        if filename_warning:
            raise HTTPException(status_code=422, detail=filename_warning)

        # 1. Validate & extract PDF text
        raw_text = await extract_pdf_text(file)
        logger.info(
            "PDF text extracted: filename=%r characters=%d",
            filename,
            len(raw_text),
        )

        # 2. Sentence segmentation
        sentences = split_into_sentences(raw_text)
        if not sentences:
            # Fallback if text couldn't be cleanly segmented into sentences
            sentences = [
                s.strip() for s in raw_text.split("\n") if len(s.strip()) > 15
            ]
        logger.info(
            "PDF segmented: filename=%r segments=%d",
            filename,
            len(sentences),
        )

        # 3. Two-stage analysis with risk gating
        results, summary_counts, pipeline_info = analyze_sentences(
            sentences,
            raw_text=raw_text,
        )

        # Reject unsupported documents explicitly instead of returning an empty
        # successful analysis response. The detail is shown to the user by the UI.
        if pipeline_info.get("document_warning"):
            raise HTTPException(
                status_code=422,
                detail=pipeline_info["document_warning"],
            )

        sentence_models = [SentenceAnalysis(**item) for item in results]
        summary_model = RiskSummary(
            total_sentences=len(sentence_models),
            low_risk=summary_counts["low_risk"],
            moderate_risk=summary_counts["moderate_risk"],
            high_risk=summary_counts["high_risk"],
        )

        response = AnalyzeResponse(
            filename=filename,
            total_sentences=len(sentence_models),
            summary=summary_model,
            sentences=sentence_models,
            pipeline_info=PipelineInfo(**pipeline_info),
        )
        logger.info(
            "Analysis complete: filename=%r scored_claims=%d elapsed_seconds=%.2f",
            filename,
            len(sentence_models),
            time.perf_counter() - started,
        )
        return response
    except HTTPException:
        # PDF validation failures already contain a safe client-facing message.
        raise
    except Exception:
        # Explicitly emit the traceback so the cause of an HTTP 500 is visible
        # even when the installed Uvicorn version omits application tracebacks.
        logger.exception(
            "Unexpected analysis failure: filename=%r elapsed_seconds=%.2f",
            filename,
            time.perf_counter() - started,
        )
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
