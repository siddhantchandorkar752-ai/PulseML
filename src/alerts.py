import logging
from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_drift_alert(drift_results: dict) -> dict:
    drift_share = drift_results.get("drift_share", 0)
    drifted = drift_share > config.DRIFT_THRESHOLD
    if drifted:
        logger.warning(f"DRIFT ALERT: drift_share={drift_share} exceeds threshold={config.DRIFT_THRESHOLD}")
    else:
        logger.info(f"No drift detected: drift_share={drift_share}")
    return {
        "alert": drifted,
        "drift_share": drift_share,
        "threshold": config.DRIFT_THRESHOLD,
        "message": "DRIFT DETECTED - Retraining recommended!" if drifted else "No drift detected.",
    }

def check_performance_alert(perf_results: dict) -> dict:
    f1_drop = perf_results.get("f1_drop", 0)
    degraded = f1_drop > config.PERFORMANCE_DROP_THRESHOLD
    if degraded:
        logger.warning(f"PERFORMANCE ALERT: f1_drop={f1_drop} exceeds threshold={config.PERFORMANCE_DROP_THRESHOLD}")
    else:
        logger.info(f"Performance stable: f1_drop={f1_drop}")
    return {
        "alert": degraded,
        "f1_drop": f1_drop,
        "threshold": config.PERFORMANCE_DROP_THRESHOLD,
        "message": "PERFORMANCE DEGRADED - Retraining recommended!" if degraded else "Performance stable.",
    }
