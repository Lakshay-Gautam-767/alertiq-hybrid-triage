import subprocess
import sys

from fastapi import APIRouter

from backend.models.schemas import FeedbackRetrainResponse
from backend.core import feedback_store
from backend.core.classifier import classifier

router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/retrain", response_model=FeedbackRetrainResponse)
def retrain():
    """
    Kicks off retraining using base data + any accumulated LLM feedback,
    then hot-reloads the classifier in memory. In a real deployment this
    would run as an async background job / scheduled pipeline, not
    synchronously inside a request — kept simple here for the demo.
    """
    n_feedback = feedback_store.count_feedback_rows()

    result = subprocess.run(
        [sys.executable, "-m", "backend.ml.train_classifier"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return FeedbackRetrainResponse(
            status="error",
            new_training_samples=n_feedback,
            message=f"Retrain failed: {result.stderr[-500:]}",
        )

    classifier.load()  # hot-reload the freshly trained artifacts

    return FeedbackRetrainResponse(
        status="success",
        new_training_samples=n_feedback,
        message="Classifier retrained and reloaded with latest feedback data.",
    )
