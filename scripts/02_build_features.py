"""Build candidate/feature tables for train and student_test and cache them.

Usage: python scripts/02_build_features.py
Outputs: cache/features_<split>.pkl (candidate rows) and cache/points_<split>.pkl
(one row per GPS point: derived label, label distance, reliability, candidate coverage).
"""
import _common  # noqa
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config
from pathsnap.data_io import list_trips
from pathsnap.prepare import prepare_trip

LABEL_MAX_DIST = 50.0  # points further than this from the reference route have unreliable labels


def one(sdir, tid):
    trip = prepare_trip(sdir, tid, with_labels=True)
    f = trip["features"]
    pts = pd.DataFrame({"trip": tid, "pt": np.arange(len(trip["xy"])), "t": trip["t"],
                        "label_arc": trip["label_arc"], "label_dist": trip["label_dist"]})
    pts["reliable"] = pts.label_dist <= LABEL_MAX_DIST
    cov = f.groupby("pt").agg(n_cand=("arc", "size"), label_in=("y", "max"))
    pts = pts.join(cov, on="pt")
    pts["n_cand"] = pts.n_cand.fillna(0).astype(int)
    pts["label_in"] = pts.label_in.fillna(0).astype(bool)
    f = f.merge(pts[["pt", "reliable"]], on="pt")
    return f, pts


if __name__ == "__main__":
    config.CACHE_DIR.mkdir(exist_ok=True)
    for split, sdir in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        t0 = time.time()
        out = Parallel(n_jobs=2)(delayed(one)(sdir, t) for t in list_trips(sdir))
        F = pd.concat([o[0] for o in out], ignore_index=True)
        P = pd.concat([o[1] for o in out], ignore_index=True)
        F.to_pickle(config.CACHE_DIR / f"features_{split}.pkl")
        P.to_pickle(config.CACHE_DIR / f"points_{split}.pkl")
        r = P[P.reliable]
        print(f"{split}: {len(P)} points, {len(F)} candidate rows, reliable {P.reliable.mean():.3%}, "
              f"no-candidate {np.mean(P.n_cand == 0):.3%}, candidate miss rate (reliable pts) {1 - r.label_in.mean():.3%}, "
              f"{time.time() - t0:.0f}s")
