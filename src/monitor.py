import logging
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import DatasetDriftMetric
from evidently.pipeline.column_mapping import ColumnMapping
from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data() -> tuple:
    reference = pd.read_csv(config.REFERENCE_DATA_PATH)
    production = pd.read_csv(config.PRODUCTION_DATA_PATH)
    return reference, production

def run_drift_report(reference: pd.DataFrame, production: pd.DataFrame) -> dict:
    logger.info("Running drift report...")
    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference, current_data=production)
    os.makedirs(config.REPORT_OUTPUT_PATH, exist_ok=True)
    report_path = os.path.join(config.REPORT_OUTPUT_PATH, "drift_report.html")
    report.save_html(report_path)
    result = report.as_dict()
    metric_result = result["metrics"][0]["result"]
    drift_share = metric_result.get("share_of_drifted_columns", metric_result.get("drift_share", 0.0))
    dataset_drifted = metric_result.get("dataset_drift", False)
    return {
        "drift_share": round(float(drift_share), 4),
        "dataset_drifted": dataset_drifted,
        "report_path": report_path,
    }

def run_performance_report(reference: pd.DataFrame, production: pd.DataFrame, model) -> dict:
    logger.info("Running performance report...")
    target = config.TARGET_COLUMN
    ref  = reference.copy()
    prod = production.copy()
    X_ref  = ref.drop(columns=[target])
    X_prod = prod.drop(columns=[target])
    ref["prediction"]  = model.predict(X_ref)
    prod["prediction"] = model.predict(X_prod)
    column_mapping = ColumnMapping(
        target=target,
        prediction="prediction",
    )
    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=ref,
        current_data=prod,
        column_mapping=column_mapping,
    )
    report_path = os.path.join(config.REPORT_OUTPUT_PATH, "performance_report.html")
    report.save_html(report_path)
    result = report.as_dict()
    metric_result = result["metrics"][0]["result"]
    ref_f1  = metric_result.get("reference", {}).get("f1", 0.0)
    curr_f1 = metric_result.get("current",   {}).get("f1", 0.0)
    return {
        "reference_f1": round(float(ref_f1),  4),
        "current_f1":   round(float(curr_f1), 4),
        "f1_drop":      round(float(ref_f1) - float(curr_f1), 4),
        "report_path":  report_path,
    }
