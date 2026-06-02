import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pangaea_readiness import evaluate

load_dotenv()

# ── App setup ────────────────────────────────────────────
app = FastAPI(
    title="PANGAEA ML Readiness API",
    description="Evaluate any PANGAEA earth science dataset for Machine Learning and Data Science suitability.",
    version="0.1.0",
)

# ── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────
class EvaluateRequest(BaseModel):
    doi: str
    use_ai: bool = True

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "doi": "10.1594/PANGAEA.787140",
                    "use_ai": True
                }
            ]
        }
    }


class EvaluateResponse(BaseModel):
    doi: str
    title: str
    shape: list
    score: int
    flag: str
    ml_tasks: list
    breakdown: dict
    warnings: list
    recommendations: list
    ai_domain: str
    ai_confidence: str
    ai_suitability: str
    ai_ml_tasks: list
    ai_ds_tasks: list
    ai_red_flags: list
    ai_opportunities: list
    error: str


# ── Routes ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "PANGAEA ML Readiness API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "evaluate": "POST /evaluate"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "saia_configured": bool(os.environ.get("SAIA_API_KEY")),
        "timestamp": time.time()
    }


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_dataset(request: EvaluateRequest):
    """
    Evaluate a PANGAEA dataset for ML/DS readiness.
    - **doi**: PANGAEA DOI or numeric ID
    - **use_ai**: Whether to run AI evaluation via SAIA/Mistral
    """
    if not request.doi.strip():
        raise HTTPException(status_code=400, detail="DOI cannot be empty")

    try:
        result = evaluate(
            request.doi.strip(),
            use_ai=request.use_ai
        )

        return EvaluateResponse(
            doi=result.doi,
            title=result.title or "",
            shape=list(result.shape),
            score=result.score,
            flag=result.flag,
            ml_tasks=result.ml_tasks,
            breakdown=result.breakdown,
            warnings=result.warnings,
            recommendations=result.recommendations,
            ai_domain=result.ai_domain or "",
            ai_confidence=result.ai_confidence or "",
            ai_suitability=result.ai_suitability or "",
            ai_ml_tasks=result.ai_ml_tasks or [],
            ai_ds_tasks=result.ai_ds_tasks or [],
            ai_red_flags=result.ai_red_flags or [],
            ai_opportunities=result.ai_opportunities or [],
            error=result.ai_error or ""
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))