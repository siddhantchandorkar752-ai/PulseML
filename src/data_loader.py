import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_raw_data() -> pd.DataFrame:
    logger.info("Loading creditcard data...")
    df = pd.read_csv("data/creditcard.csv", encoding_errors="replace")
    logger.info(f"Loaded shape: {df.shape}")
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing data...")
    df = df[df.columns[df.isnull().mean() < 0.5]]
    df = df.select_dtypes(include=["number"])
    df = df.fillna(df.median())
    logger.info(f"Preprocessed shape: {df.shape}")
    return df

def split_reference_production(df: pd.DataFrame) -> tuple:
    target = config.TARGET_COLUMN
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    reference, production = train_test_split(
        df, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[target],
    )
    production_sample = production.sample(
        n=min(config.PRODUCTION_SAMPLE_SIZE, len(production)),
        random_state=config.RANDOM_STATE,
    ).copy()
    numeric_cols = [c for c in production_sample.columns if c != target]
    production_sample[numeric_cols] = production_sample[numeric_cols] * 1.15
    logger.info(f"Reference shape: {reference.shape}")
    logger.info(f"Production shape: {production_sample.shape}")
    return reference, production_sample

def save_splits(reference: pd.DataFrame, production: pd.DataFrame) -> None:
    reference.to_csv(config.REFERENCE_DATA_PATH, index=False)
    production.to_csv(config.PRODUCTION_DATA_PATH, index=False)
    logger.info(f"Saved reference -> {config.REFERENCE_DATA_PATH}")
    logger.info(f"Saved production -> {config.PRODUCTION_DATA_PATH}")

def prepare_data() -> tuple:
    df = load_raw_data()
    df = preprocess(df)
    reference, production = split_reference_production(df)
    save_splits(reference, production)
    return reference, production
