
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
    REFERENCE_DATA_PATH: str = os.getenv("REFERENCE_DATA_PATH", "data/reference.csv")
    PRODUCTION_DATA_PATH: str = os.getenv("PRODUCTION_DATA_PATH", "data/production.csv")
    REPORT_OUTPUT_PATH: str = os.getenv("REPORT_OUTPUT_PATH", "reports/")

    # Thresholds
    DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", 0.15))
    PERFORMANCE_DROP_THRESHOLD: float = float(os.getenv("PERFORMANCE_DROP_THRESHOLD", 0.05))

    # Model
    TARGET_COLUMN: str = "isFraud"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    PRODUCTION_SAMPLE_SIZE: int = 10000

    # XGBoost
    XGB_PARAMS: dict = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": 10,
        "random_state": 42,
        "eval_metric": "auc",
    }

config = Config()
