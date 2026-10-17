from typing import List, Optional
from pydantic import BaseModel, Field


class SentenceAnalysis(BaseModel):
    id: int
    text: str
    risk: str = Field(..., description="Risk tier: Low, Moderate, or High")
    score: float = Field(..., description="Confidence score between 0.0 and 1.0")
    question: Optional[str] = Field(
        None,
        description="Targeted investor counter-question for Moderate/High risk sentences, None for Low risk",
    )


class RiskSummary(BaseModel):
    total_sentences: int
    low_risk: int
    moderate_risk: int
    high_risk: int


class PipelineInfo(BaseModel):
    classifier: str
    generator: str
    mode: str
    note: str


class AnalyzeResponse(BaseModel):
    filename: str
    total_sentences: int
    summary: RiskSummary
    sentences: List[SentenceAnalysis]
    pipeline_info: PipelineInfo


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    generator_loaded: bool
    device: str
    version: str = "0.1.0"
