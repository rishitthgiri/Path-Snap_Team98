"""PathSnap FastAPI backend (runs on Laptop 1).

Run:   PATHSNAP_EVAL_DATA=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
Endpoints
  GET  /health          -> status + loaded artifacts
  GET  /trips           -> trip IDs available in the evaluation folder (no labels)
  POST /predict         -> {"trip_id": "..."}; runs features -> classifier -> HMM/Viterbi
                           -> route, then scores it server-side against the reference route.
Ground truth (.route) is read only inside evaluate_prediction(), after prediction, and
is never returned to the client: the response holds the predicted route + metrics only.
"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import config
from pathsnap.data_io import list_trips
from pathsnap.inference import Predictor, evaluate_prediction, to_geojson_like

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

EVAL_DIR = config.EVAL_DIR if config.EVAL_DIR.exists() else config.STUDENT_TEST_DIR
predictor = Predictor()
TRIPS = list_trips(EVAL_DIR)
log.info("Loaded classifier + HMM config. Serving %d trips from %s", len(TRIPS), EVAL_DIR)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "eval_dir": str(EVAL_DIR), "n_trips": len(TRIPS),
            "hmm_params": predictor.hmm_params}


@app.get("/trips")
def trips():
    return {"trips": TRIPS, "source": EVAL_DIR.name}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id.strip()
    client = request.client.host if request.client else "?"
    log.info("REQUEST  /predict trip=%s from %s", tid, client)
    if tid not in TRIPS:
        log.warning("unknown trip %s", tid)
        raise HTTPException(404, f"unknown trip_id {tid}")
    t0 = time.time()
    pred = predictor.predict_trip(EVAL_DIR, tid)
    metrics = evaluate_prediction(EVAL_DIR, tid, pred)
    geo = to_geojson_like(pred)
    total = time.time() - t0
    log.info("RESULT   trip=%s points=%d route_arcs=%d | point_acc HMM=%.4f clf=%.4f | route F1=%.4f "
             "P=%.4f R=%.4f | continuity=%.3f | %.2fs (features %.2fs, clf %.2fs, viterbi %.2fs)",
             tid, metrics["n_points"], len(pred["route"]), metrics["hmm_point_accuracy"],
             metrics["classifier_only_point_accuracy"], metrics["route_f1"], metrics["route_precision"],
             metrics["route_recall"], metrics["hmm_route_continuity"], total,
             pred["timing_s"]["features"], pred["timing_s"]["classifier"], pred["timing_s"]["viterbi"])
    return {"trip_id": tid, "metrics": metrics, "timing_s": {**pred["timing_s"], "total_incl_scoring": total},
            "n_bridged_gaps": pred["n_bridged"], "n_unbridged_gaps": pred["n_unbridged"], **geo}
