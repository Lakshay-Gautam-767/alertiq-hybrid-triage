"""
Classical ML layer: TF-IDF + Logistic Regression.
This is the FAST, CHEAP path that should handle the majority of routine,
already-seen log/ticket patterns. Only uncertain cases get escalated.
"""
import joblib
from pathlib import Path
from typing import Tuple

from backend import config


class ClassicalClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self._loaded = False

    def load(self) -> bool:
        """Load trained artifacts from disk. Returns False if not trained yet."""
        if not (config.MODEL_PATH.exists() and config.VECTORIZER_PATH.exists()
                and config.LABEL_ENCODER_PATH.exists()):
            self._loaded = False
            return False

        self.model = joblib.load(config.MODEL_PATH)
        self.vectorizer = joblib.load(config.VECTORIZER_PATH)
        self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
        self._loaded = True
        return True

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def predict(self, text: str) -> Tuple[str, float]:
        """Returns (predicted_label, confidence[0-1])."""
        if not self._loaded:
            raise RuntimeError(
                "Classifier not trained yet. Run: python -m backend.ml.train_classifier"
            )
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        best_idx = probs.argmax()
        label = self.label_encoder.inverse_transform([best_idx])[0]
        confidence = float(probs[best_idx])
        return label, confidence


# Singleton instance used across the app (loaded once at startup)
classifier = ClassicalClassifier()
