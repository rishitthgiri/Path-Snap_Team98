"""Tune HMM decoder parameters and compare classifier-only vs classifier+HMM.

Tuning set: 16 training trips whose probabilities come from out-of-fold models
(HGB with the BO-selected parameters, trained on the OTHER training trips), so the
decoder is tuned on realistic, not over-confident, probabilities. Student-test
trips are then used for the reported comparison. Evaluator data is never used.

Usage: python scripts/05_hmm_eval.py
Outputs: artifacts/inference_config.json, results/hmm_grid.csv, results/hmm_vs_classifier_*.csv, plots/hmm_*.png
"""
import _common  # noqa
import itertools, json, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.data_io import load_trip, clean_track, list_trips
from pathsnap.network import build_trip
from pathsnap.hmm import viterbi_decode, reconstruct_route
from pathsnap.metrics import route_continuity, route_prf, collapse
from pathsnap.models import MODEL_SPECS

C = config.CACHE_DIR
Ftr = pd.read_pickle(C / "features_train.pkl"); Ptr = pd.read_pickle(C / "points_train.pkl")
Fte = pd.read_pickle(C / "features_student_test.pkl"); Pte = pd.read_pickle(C / "points_student_test.pkl")
best = json.load(open(config.RESULTS_DIR / "bayes_opt_summary.json"))["best_params"]

# ---- out-of-fold probabilities for 16 tuning trips ----
sizes = Ptr.groupby("trip").size()
rng = np.random.default_rng(config.SEED)
tune_trips = sorted(rng.choice(sizes[sizes < 4000].index.values, 16, replace=False))
halves = [tune_trips[::2], tune_trips[1::2]]
Ftune = []
for h in halves:
    trn = Ftr[Ftr.reliable & ~Ftr.trip.isin(h) & (Ftr.pt % config.TRAIN_POINT_STRIDE == 0)]
    m = HistGradientBoostingClassifier(max_iter=500, early_stopping=True, n_iter_no_change=20,
                                       random_state=config.SEED, **best).fit(trn[FEATURES].values, trn.y.values)
    f = Ftr[Ftr.trip.isin(h)].copy(); f["prob"] = m.predict_proba(f[FEATURES].values)[:, 1]; Ftune.append(f)
Ftune = pd.concat(Ftune)


def load(split_dir, tid):
    tr = load_trip(split_dir, tid); tr["track"], _ = clean_track(tr["track"]); return build_trip(tr)


def score(trip, cand, pts, params):
    dec = viterbi_decode(trip, cand, params)
    rel = pts[pts.reliable]
    return float(np.mean(dec[rel.pt.values] == rel.label_arc.values)), dec


def eval_grid_trip(tid, combos):
    trip = load(config.TRAIN_DIR, tid)
    cand = Ftune[Ftune.trip == tid]; pts = Ptr[Ptr.trip == tid]
    return [(tid, *c, score(trip, cand, pts, dict(beta=c[0], emission_weight=c[1], back_tol=c[2]))[0], int(pts.reliable.sum())) for c in combos]


combos = list(itertools.product([3, 6, 10, 20, 40], [0.5, 1.0, 2.0], [5, 15, 30]))
t0 = time.time()
out = Parallel(n_jobs=2)(delayed(eval_grid_trip)(t, combos) for t in tune_trips)
G = pd.DataFrame([r for o in out for r in o], columns=["trip", "beta", "emission_weight", "back_tol", "acc", "n"])
G["w"] = G.acc * G.n
grid = G.groupby(["beta", "emission_weight", "back_tol"]).apply(lambda g: g.w.sum() / g.n.sum(), include_groups=False).rename("tune_point_acc").reset_index()
grid.to_csv(config.RESULTS_DIR / "hmm_grid.csv", index=False)
bestrow = grid.loc[grid.tune_point_acc.idxmax()]
hmm_params = dict(config.HMM_DEFAULTS, beta=float(bestrow.beta), emission_weight=float(bestrow.emission_weight), back_tol=float(bestrow.back_tol))
print(f"grid search {len(combos)} combos x {len(tune_trips)} trips in {time.time() - t0:.0f}s; best {hmm_params} acc {bestrow.tune_point_acc:.4f}")
json.dump({"features": FEATURES, "hmm_params": hmm_params, "classifier": "HistGradientBoosting (BO-tuned)",
           "candidate_radius_m": config.CAND_RADIUS_M, "candidate_max_k": config.CAND_MAX_K,
           "tune_trips": [str(t) for t in tune_trips]},
          open(config.ARTIFACT_DIR / "inference_config.json", "w"), indent=2)

# ---- student test: classifier-only vs HMM for the final model and for every compared model ----
final = joblib.load(config.ARTIFACT_DIR / "classifier.joblib")
Fte_r = Fte.copy()
probs = {"HGB tuned (final)": final.predict_proba(Fte_r[FEATURES].values)[:, 1]}
rel_mask = Fte_r.reliable.values
for name in MODEL_SPECS:
    f = C / f"pte_{name}.npy"
    if f.exists():
        p = np.full(len(Fte_r), np.nan); p[rel_mask] = np.load(f)
        # unreliable points were not scored by 03; give them the final model's prob
        p[~rel_mask] = probs["HGB tuned (final)"][~rel_mask]; probs[name] = p

trips = {t: load(config.STUDENT_TEST_DIR, t) for t in list_trips(config.STUDENT_TEST_DIR)}
rows = []
for mname, p in probs.items():
    Fte_r["prob"] = p
    for tid, trip in trips.items():
        cand = Fte_r[Fte_r.trip == tid]; pts = Pte[Pte.trip == tid]; rel = pts[pts.reliable]
        idx = cand.groupby("pt").prob.idxmax()
        clf = np.full(len(trip["xy"]), -1); clf[cand.loc[idx, "pt"].values] = cand.loc[idx, "arc"].values
        t1 = time.time(); dec = viterbi_decode(trip, cand, hmm_params); vt = time.time() - t1
        route, nb, nu = reconstruct_route(dec, trip["net"])
        L, A = trip["net"].length, trip["arcs"]
        r = {"model": mname, "trip": tid, "n_points": len(rel),
             "clf_top1": np.mean(clf[rel.pt] == rel.label_arc), "hmm_top1": np.mean(dec[rel.pt] == rel.label_arc),
             "clf_continuity": route_continuity(clf, A), "hmm_continuity": route_continuity(dec, A),
             "label_continuity": route_continuity(pts.label_arc.values, A),
             "final_route_continuity": route_continuity(route, A), "viterbi_s": vt, "n_bridged": nb, "n_unbridged": nu}
        r.update({"clf_" + k: v for k, v in route_prf(collapse(clf), trip["route"], L).items()})
        r.update({"hmm_" + k: v for k, v in route_prf(route, trip["route"], L).items()})
        rows.append(r)
    print(mname, "done")
R = pd.DataFrame(rows)
R.to_csv(config.RESULTS_DIR / "hmm_vs_classifier_per_trip.csv", index=False)
agg = R.groupby("model").apply(lambda g: pd.Series({
    "clf_top1": np.average(g.clf_top1, weights=g.n_points), "hmm_top1": np.average(g.hmm_top1, weights=g.n_points),
    "clf_continuity": g.clf_continuity.mean(), "hmm_continuity": g.hmm_continuity.mean(),
    "label_continuity": g.label_continuity.mean(), "final_route_continuity": g.final_route_continuity.mean(),
    "clf_route_f1": g.clf_route_f1.mean(), "hmm_route_f1": g.hmm_route_f1.mean(),
    "hmm_route_precision": g.hmm_route_precision.mean(), "hmm_route_recall": g.hmm_route_recall.mean(),
    "trips_exact_route_f1_gt_0.95": int((g.hmm_route_f1 > .95).sum())}), include_groups=False).reset_index()
agg.to_csv(config.RESULTS_DIR / "hmm_vs_classifier_summary.csv", index=False)
print(agg.round(4).to_string(index=False))

fin = R[R.model == "HGB tuned (final)"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5)); x = np.arange(len(fin))
ax[0].bar(x - .2, fin.clf_top1, .4, label="classifier only"); ax[0].bar(x + .2, fin.hmm_top1, .4, label="classifier + HMM")
ax[0].set_xticks(x, fin.trip, rotation=45); ax[0].set(title="Student test: observation Top-1 per trip", ylim=(.6, 1)); ax[0].legend()
ax[1].bar(x - .27, fin.clf_continuity, .27, label="classifier only"); ax[1].bar(x, fin.hmm_continuity, .27, label="classifier + HMM")
ax[1].bar(x + .27, fin.label_continuity, .27, label="reference labels", color="grey")
ax[1].set_xticks(x, fin.trip, rotation=45); ax[1].set(title="Route continuity (adjacent-arc rate)", ylim=(.3, 1.02)); ax[1].legend()
plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "hmm_vs_classifier.png", dpi=120); plt.close()
pv = grid[grid.back_tol == bestrow.back_tol].pivot(index="beta", columns="emission_weight", values="tune_point_acc")
fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(pv.values, cmap="viridis")
ax.set_xticks(range(len(pv.columns)), pv.columns); ax.set_yticks(range(len(pv.index)), pv.index)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]): ax.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
ax.set(xlabel="emission weight", ylabel="beta (m)", title=f"HMM grid (back_tol={bestrow.back_tol:g})"); plt.colorbar(im)
plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "hmm_grid.png", dpi=120); plt.close()
