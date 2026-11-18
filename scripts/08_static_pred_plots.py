"""Static (PNG) version of the prediction maps for the report: GPS, network, reference, predicted.

Usage: python scripts/08_static_pred_plots.py --trips 00000009,00000077
"""
import _common  # noqa
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathsnap import config
from pathsnap.inference import Predictor, evaluate_prediction

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="student_test"); ap.add_argument("--trips", default="00000009,00000077")
a = ap.parse_args(); sdir = config.DATA_DIR / a.split; P = Predictor()
for tid in a.trips.split(","):
    pred = P.predict_trip(sdir, tid); m = evaluate_prediction(sdir, tid, pred)
    tr = pred["trip"]; nd, ar, trk = tr["nodes"], tr["arcs"], tr["track"]
    from pathsnap.data_io import load_trip
    ref = load_trip(sdir, tid)["route"]
    lo, hi = trk[:, :2].min(0) - .002, trk[:, :2].max(0) + .002
    ins = ((nd >= lo) & (nd <= hi)).all(1); sel = ins[ar[:, 0]] & ins[ar[:, 1]]
    seg = lambda ids: np.stack([nd[ar[ids, 0]], nd[ar[ids, 1]]], 1)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(LineCollection(seg(np.flatnonzero(sel)), colors="#c8c8c8", lw=.5, label="road network"))
    ax.add_collection(LineCollection(seg(ref), colors="#2e7d32", lw=5, alpha=.5, label="reference route"))
    ax.add_collection(LineCollection(seg(np.array(pred["route"])), colors="#d32f2f", lw=1.6, label="predicted route (HMM)"))
    ax.scatter(trk[:, 0], trk[:, 1], s=2, c="#1565c0", label="raw GPS", zorder=3)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect(1 / np.cos(np.radians(trk[:, 1].mean())))
    ax.legend(loc="upper right"); ax.set(xlabel="longitude", ylabel="latitude",
        title=f"{a.split} {tid}: point acc {m['hmm_point_accuracy']:.3f} (clf-only {m['classifier_only_point_accuracy']:.3f}), route F1 {m['route_f1']:.3f}")
    plt.tight_layout(); plt.savefig(config.PLOTS_DIR / f"pred_{a.split}_{tid}.png", dpi=120); plt.close(); print("wrote", tid)
