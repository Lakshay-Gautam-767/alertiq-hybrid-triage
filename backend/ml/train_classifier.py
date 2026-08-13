"""
Trains the classical TF-IDF + Logistic Regression classifier on
backend/ml/data/sample_logs.csv and saves artifacts to backend/ml/artifacts/.

Run this:
    python -m backend.ml.train_classifier

Re-run it any time after new feedback data has accumulated
(see api/routes_feedback.py -> /api/retrain) to close the feedback loop.
"""
import csv
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from backend import config
from backend.ml.generate_sample_data import generate as generate_sample_data


def load_training_data() -> pd.DataFrame:
    if not config.TRAIN_DATA_PATH.exists():
        print("No base training data found — generating synthetic sample dataset...")
        generate_sample_data()

    df = pd.read_csv(config.TRAIN_DATA_PATH)

    # Fold in any accumulated feedback from LLM escalations (the "feedback loop")
    if config.FEEDBACK_DATA_PATH.exists():
        feedback_df = pd.read_csv(config.FEEDBACK_DATA_PATH)
        if not feedback_df.empty:
            print(f"Including {len(feedback_df)} feedback rows from LLM escalations")
            df = pd.concat([df, feedback_df], ignore_index=True)

    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    return df


def train():
    df = load_training_data()
    print(f"Training on {len(df)} total samples across labels: {df['label'].unique().tolist()}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(df["text"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n--- Evaluation on held-out test set ---")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)
    print(f"\nSaved model artifacts -> {config.MODEL_DIR}")


if __name__ == "__main__":
    train()
