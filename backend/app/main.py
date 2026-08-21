"""HTTP API to make request to the backend"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model.inference import BACKEND_ROOT, GenerationSettings, get_reviewer
from app.services.review import review_diff

logger = logging.getLogger(__name__)

MAX_DIFF_BYTES = 2_000_000


class ReviewRequest(BaseModel):
    diff: str = Field(..., description="Unified diff to review.")
    max_comments: int | None = Field(
        default=20,
        ge=1,
        le=200,
        description="Cap on comments returned; each one costs a model call.",
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class CommentResponse(BaseModel):
    path: str
    line: int
    body: str
    hunk_header: str


class ReviewResponse(BaseModel):
    comments: list[CommentResponse]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(BACKEND_ROOT / ".env")
    reviewer = get_reviewer()
    try:
        reviewer.load()
        logger.info("Model loaded on GPU %s", reviewer.gpu_index)
    except Exception:
        # Let /health report the failure rather than crashing the whole process,
        # so the container can start on a machine without a GPU.
        logger.exception("Model failed to load; /review will return 503")
    yield


app = FastAPI(title="Sentinel", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    reviewer = get_reviewer()
    return {"status": "ok", "model_loaded": reviewer.is_loaded}


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest) -> ReviewResponse:
    if not request.diff.strip():
        raise HTTPException(status_code=422, detail="diff is empty")
    if len(request.diff.encode()) > MAX_DIFF_BYTES:
        raise HTTPException(status_code=413, detail="diff too large")

    reviewer = get_reviewer()
    if not reviewer.is_loaded:
        raise HTTPException(status_code=503, detail="model not loaded")

    settings = GenerationSettings()
    if request.temperature is not None:
        settings = GenerationSettings(
            max_new_tokens=settings.max_new_tokens,
            temperature=request.temperature,
            top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty,
        )

    comments = review_diff(
        request.diff,
        settings=settings,
        max_comments=request.max_comments,
    )
    return ReviewResponse(
        comments=[CommentResponse(**vars(c)) for c in comments]
    )
