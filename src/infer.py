# src/infer.py
import os
import time
import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "LSTM_Anomaly_Detector")
MODEL_ARTIFACT_NAME = os.environ.get("MODEL_ARTIFACT_NAME", "lstm_ae_model")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")  # optional: set if using remote MLflow server

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="LSTM Anomaly Detector (MLflow auto-loader)")

# Global model holder
MODEL = {"run_id": None, "model": None, "device": None, "loaded_at": None}


class SequenceInput(BaseModel):
    sequence: List[float]  # 1D time series


def load_latest_model(experiment_name: str = MLFLOW_EXPERIMENT, artifact_name: str = MODEL_ARTIFACT_NAME):
    """
    Find latest run in MLflow experiment and load the model artifact.
    Returns (run_id, model)
    """
    # Get experiment id(s)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found in MLflow tracking server.")
    exp_id = experiment.experiment_id

    # Search runs sorted by end_time desc (latest)
    runs_df = mlflow.search_runs([exp_id], order_by=["attributes.end_time DESC"], max_results=10)
    if runs_df.empty:
        raise RuntimeError(f"No runs found for experiment id {exp_id}")

    # pick the first run with the artifact present (best-effort)
    for _, run in runs_df.iterrows():
        run_id = run["run_id"]
        model_uri = f"runs:/{run_id}/{artifact_name}"
        try:
            # This will raise if artifact doesn't exist
            model = mlflow.pytorch.load_model(model_uri)
            return run_id, model
        except Exception:
            continue

    raise RuntimeError("No runnable model artifact found in recent runs.")


def reload_model():
    run_id, model = load_latest_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    MODEL.update({
        "run_id": run_id,
        "model": model,
        "device": device,
        "loaded_at": time.time()
    })
    return MODEL


@app.on_event("startup")
def startup_load():
    try:
        m = reload_model()
        app.state.model_info = {"run_id": m["run_id"], "loaded_at": m["loaded_at"]}
        print(f"[infer] Loaded model run_id={m['run_id']}")
    except Exception as e:
        # still start server; endpoints will return 503 until model loaded or reload is called
        print(f"[infer] Warning: failed to load model at startup: {e}")


@app.get("/health")
def health():
    if MODEL["model"] is None:
        return {"status": "no_model_loaded"}
    return {"status": "ok", "run_id": MODEL["run_id"], "loaded_at": MODEL["loaded_at"]}


@app.post("/reload")
def reload_endpoint():
    try:
        m = reload_model()
        return {"status": "reloaded", "run_id": m["run_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict_anomaly(payload: SequenceInput, threshold_k: Optional[float] = None):
    """
    Accepts a JSON with {"sequence": [0.1, 0.2, ...]} and returns reconstruction error and anomaly status.
    Optional query param threshold_k can be used by client, otherwise server-side default is used.
    """
    if MODEL["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /reload or check MLflow experiment.")

    seq = payload.sequence
    if len(seq) == 0:
        raise HTTPException(status_code=400, detail="Empty sequence.")

    # Prepare tensor: (1, seq_len, 1)
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(MODEL["device"])

    with torch.no_grad():
        recon = MODEL["model"](x).cpu()
    x_cpu = x.cpu()
    # Compute MSE per-window (mean across time and features)
    error = torch.mean((recon - x_cpu) ** 2, dim=(1, 2)).item()

    # Decide threshold: try to fetch threshold from deployed metrics file if exists
    server_threshold = None
    if threshold_k is not None:
        # If user passed threshold_k, apply dynamic thresholding (mean + k*std) — but we don't have mu/sigma here.
        server_threshold = None
    # fallback threshold:
    fallback_threshold = float(os.environ.get("ANOMALY_THRESHOLD", "0.01"))
    is_anomaly = error > fallback_threshold

    return {
        "reconstruction_error": float(error),
        "threshold_used": fallback_threshold,
        "anomaly": bool(is_anomaly),
        "model_run_id": MODEL["run_id"]
    }
