import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)


def load_or_generate(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)

    logger.info("Data not found; generating dummy dataset.")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 4))
    y = rng.integers(0, 2, size=(200,))
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["target"] = y
    # Save the generated dataset so subsequent runs use it
    df.to_csv(path, index=False)
    logger.info(f"Dummy data written to {path}")
    return df


def train_and_save(df: pd.DataFrame, model_path: str):
    X = df.drop(columns=["target"]) if "target" in df.columns else df.iloc[:, :-1]
    y = df["target"] if "target" in df.columns else df.iloc[:, -1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Training RandomForestClassifier (lightweight)...")
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    logger.info(f"Validation accuracy: {acc:.4f}")

    joblib.dump(clf, model_path)
    logger.info(f"Saved best model to {model_path}")


def main():
    ensure_dirs()
    data_path = os.path.join("data", "train.csv")
    df = load_or_generate(data_path)
    model_path = os.path.join("models", "best_model.joblib")
    train_and_save(df, model_path)


if __name__ == "__main__":
    main()
