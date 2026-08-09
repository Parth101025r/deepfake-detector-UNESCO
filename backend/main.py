import os
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline import MultimodalRAGPipeline, PROJECT_ROOT
from backend.schemas import VerifyResponse


app = FastAPI(
    title="Multimodal Fake News Detection API",
    description="MMFakeBench-style text + image verification with CLIP and local FAISS retrieval.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_pipeline():
    return MultimodalRAGPipeline()


@app.get("/health")
def health_check():
    if get_pipeline.cache_info().currsize:
        pipeline = get_pipeline()
        return {
            "status": "ok",
            "pipeline_loaded": True,
            "checkpoint_loaded": pipeline.checkpoint_loaded,
            "retrieval_ready": bool(
                pipeline.retriever.ready
                or pipeline.gemini_verifier.ready
                or pipeline.fact_check_retriever.ready
                or pipeline.news_retriever.ready
                or pipeline.knowledge_retriever.ready
                or pipeline.general_knowledge_retriever.ready
            ),
            "local_retrieval_ready": pipeline.retriever.ready,
            "gemini_ready": pipeline.gemini_verifier.ready,
            "fact_check_ready": pipeline.fact_check_retriever.ready,
            "live_news_ready": pipeline.news_retriever.ready,
            "trusted_knowledge_ready": pipeline.knowledge_retriever.ready,
            "general_knowledge_ready": pipeline.general_knowledge_retriever.ready,
            "fact_check_error": (
                pipeline.gemini_verifier.last_error
                or pipeline.fact_check_retriever.last_error
                or pipeline.news_retriever.last_error
                or pipeline.knowledge_retriever.last_error
                or pipeline.general_knowledge_retriever.last_error
            ),
        }

    local_retrieval_ready = (
        (PROJECT_ROOT / "retrieval" / "index.faiss").exists()
        and (PROJECT_ROOT / "retrieval" / "metadata.json").exists()
    )
    fact_check_ready = bool(os.getenv("GOOGLE_FACT_CHECK_API_KEY"))
    gemini_ready = bool(os.getenv("GEMINI_API_KEY")) or (PROJECT_ROOT / "secrets" / "gemini_api_key.txt").exists()
    return {
        "status": "ok",
        "pipeline_loaded": False,
        "checkpoint_loaded": (PROJECT_ROOT / "checkpoints" / "model_best.pt").exists(),
        "retrieval_ready": True,
        "local_retrieval_ready": local_retrieval_ready,
        "gemini_ready": gemini_ready,
        "fact_check_ready": fact_check_ready,
        "live_news_ready": True,
        "trusted_knowledge_ready": True,
        "general_knowledge_ready": True,
        "fact_check_error": None,
    }


@app.post("/verify", response_model=VerifyResponse)
async def verify(
    claim: str = Form(..., description="The text claim to verify"),
    top_k: int = Form(3, description="Number of evidence snippets to retrieve"),
    image: UploadFile = File(None, description="Optional image payload"),
):
    try:
        image_bytes = None
        if image and image.filename:
            image_bytes = await image.read()

        result = get_pipeline().predict(
            claim=claim,
            image_bytes=image_bytes,
            top_k=top_k,
        )
        result["status"] = "success"
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
