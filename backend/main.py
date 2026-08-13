from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend import config
from backend.core.classifier import classifier
from backend.api import routes_logs, routes_stats, routes_feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = classifier.load()
    if not loaded:
        print(
            "\n[WARNING] No trained model found.\n"
            "Run this before using the API: python -m backend.ml.train_classifier\n"
        )
    else:
        print("[OK] Classical classifier loaded successfully.")
    yield


app = FastAPI(
    title="Hybrid Log/Anomaly Classification System",
    description="Classical ML for routine events, LLM escalation for ambiguous/critical ones.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_logs.router)
app.include_router(routes_stats.router)
app.include_router(routes_feedback.router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "model_ready": classifier.is_ready,
        "docs": "/docs",
    }
