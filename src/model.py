import logging
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_features_target(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    return X, y

def train_model(reference: pd.DataFrame) -> XGBClassifier:
    logger.info("Training XGBoost model...")
    X, y = get_features_target(reference)
    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(X, y, verbose=False)
    logger.info("Training complete.")
    return model

def evaluate_model(model: XGBClassifier, df: pd.DataFrame) -> dict:
    X, y = get_features_target(df)
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y, preds), 4),
        "f1_score": round(f1_score(y, preds), 4),
        "precision": round(precision_score(y, preds), 4),
        "recall": round(recall_score(y, preds), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }
    logger.info(f"Metrics: {metrics}")
    return metrics

def save_model(model: XGBClassifier) -> None:
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved -> {config.MODEL_PATH}")

def load_model() -> XGBClassifier:
    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded <- {config.MODEL_PATH}")
    return model
