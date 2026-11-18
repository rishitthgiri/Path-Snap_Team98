"""End-to-end prediction for one trip (used by the API and offline scripts).

predict_trip() never reads the .route file. Reference data is only touched in
evaluate_prediction(), after the prediction is finished, so ground truth can never
leak into the model or the decoder.
"""
import json
import time
import numpy as np
import joblib

from . import config
from .prepare import prepare_trip
from .features import FEATURES
from .hmm import viterbi_decode, reconstruct_route
from .data_io import load_trip, clean_track
from .labels import align_points_to_route
from .metrics import route_continuity, route_prf, collapse


class Predictor:
    def __init__(self, artifact_dir=None):
        d = artifact_dir or config.ARTIFACT_DIR
        self.model = joblib.load(d / "classifier.joblib")
        meta = json.load(open(d / "inference_config.json"))
        self.features = meta["features"]
        self.hmm_params = meta["hmm_params"]
        assert self.features == FEATURES, "feature list changed since training"

    def predict_trip(self, split_dir, trip_id):
        t0 = time.time()
        trip = prepare_trip(split_dir, trip_id, with_labels=False)
        t1 = time.time()
        cand = trip["features"]
        if len(cand):
            cand = cand.copy()
            cand["prob"] = self.model.predict_proba(cand[self.features].values)[:, 1]
            clf_idx = cand.groupby("pt")["prob"].idxmax()
            clf_only = np.full(len(trip["xy"]), -1, dtype=np.int64)
            clf_only[cand.loc[clf_idx, "pt"].values] = cand.loc[clf_idx, "arc"].values
        else:
            clf_only = np.full(len(trip["xy"]), -1, dtype=np.int64)
        t2 = time.time()
        decoded = viterbi_decode(trip, cand, self.hmm_params) if len(cand) else clf_only
        t3 = time.time()
        route, n_bridged, n_broken = reconstruct_route(decoded, trip["net"])
        t4 = time.time()
        return {"trip": trip, "decoded": decoded, "clf_only": clf_only, "route": route,
                "n_bridged": n_bridged, "n_unbridged": n_broken,
                "timing_s": {"features": t1 - t0, "classifier": t2 - t1, "viterbi": t3 - t2,
                             "route_build": t4 - t3, "total": t4 - t0}}


def evaluate_prediction(split_dir, trip_id, pred, label_max_dist=50.0):
    """Server-side scoring against the reference route (never sent to the model/UI)."""
    trip = pred["trip"]
    ref = load_trip(split_dir, trip_id)["route"]
    if ref is None:
        return None
    label_arc, label_dist, _ = align_points_to_route(trip["xy"], trip["net"], ref)
    ok = label_dist <= label_max_dist
    arcs, L = trip["arcs"], trip["net"].length
    out = {
        "n_points": int(len(ok)), "n_scored_points": int(ok.sum()),
        "hmm_point_accuracy": float(np.mean(pred["decoded"][ok] == label_arc[ok])),
        "classifier_only_point_accuracy": float(np.mean(pred["clf_only"][ok] == label_arc[ok])),
        "hmm_route_continuity": route_continuity(pred["decoded"], arcs),
        "classifier_only_route_continuity": route_continuity(pred["clf_only"], arcs),
        "reference_label_continuity": route_continuity(label_arc, arcs),
        "final_route_continuity": route_continuity(pred["route"], arcs),
    }
    out.update(route_prf(pred["route"], ref, L))
    out.update({"clf_only_" + k: v for k, v in route_prf(collapse(pred["clf_only"]), ref, L).items()})
    return out


def to_geojson_like(pred, max_points=2000):
    trip = pred["trip"]
    nd, ar, tr = trip["nodes"], trip["arcs"], trip["track"]
    step = max(1, len(tr) // max_points)
    gps = [[float(lat), float(lon)] for lon, lat in tr[::step, :2]]
    segs = [[[float(nd[a, 1]), float(nd[a, 0])], [float(nd[b, 1]), float(nd[b, 0])]] for a, b in ar[pred["route"]]]
    return {"gps_points": gps, "route_segments": segs, "route_arc_ids": [int(a) for a in pred["route"]]}
