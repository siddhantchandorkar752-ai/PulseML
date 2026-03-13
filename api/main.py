import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from fastapi import FastAPI, HTTPException
from src.data_loader import prepare_data
from src.model import train_model, evaluate_model, save_model, load_model
from src.monitor import run_drift_report, run_performance_report, load_data
from src.alerts import check_drift_alert, check_performance_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PulseML API", description="ML Monitoring & Drift Detection", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "PulseML"}

@app.post("/prepare")
def prepare():
    try:
        reference, production = prepare_data()
        return {"status": "success", "reference_rows": len(reference), "production_rows": len(production)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train():
    try:
        reference, _ = load_data()
        model = train_model(reference)
        metrics = evaluate_model(model, reference)
        save_model(model)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift")
def drift():
    try:
        reference, production = load_data()
        drift_results = run_drift_report(reference, production)
        alert = check_drift_alert(drift_results)
        return {"drift": drift_results, "alert": alert}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance")
def performance():
    try:
        reference, production = load_data()
        model = load_model()
        perf_results = run_performance_report(reference, production, model)
        alert = check_performance_alert(perf_results)
        return {"performance": perf_results, "alert": alert}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitor")
def monitor():
    try:
        reference, production = load_data()
        model = load_model()
        drift_results = run_drift_report(reference, production)
        perf_results = run_performance_report(reference, production, model)
        drift_alert = check_drift_alert(drift_results)
        perf_alert = check_performance_alert(perf_results)
        return {
            "drift": drift_results,
            "performance": perf_results,
            "alerts": {
                "drift_alert": drift_alert,
                "performance_alert": perf_alert,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
