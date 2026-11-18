"""Train and compare candidate classifiers.

Protocol
- Fit/CV on the 80 training trips only, GroupKFold(5) by trip (no trip in two folds).
- Rows: reliable points only, every TRAIN_POINT_STRIDE-th point (1 Hz points are near-duplicates).
- Development comparison on the 10 student-test trips (all reliable points).
- Metrics: binary Macro F1 / precision / recall / log loss on candidate rows,
  observation-level Top-1 accuracy (argmax per GPS point), candidate miss rate,
  route continuity of classifier-only argmax route, train / inference time.

Usage: python scripts/03_train_compare.py [--models A,B,...]
"""
import _common  # noqa
import argparse, json, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix, ConfusionMatrixDisplay

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.models import MODEL_SPECS
from pathsnap.metrics import top1_choice, observation_top1, route_continuity
from pathsnap.data_io import list_trips

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=",".join(MODEL_SPECS))
args = ap.parse_args()

C = config.CACHE_DIR
Ftr = pd.read_pickle(C / "features_train.pkl"); Ptr = pd.read_pickle(C / "points_train.pkl")
Fte = pd.read_pickle(C / "features_student_test.pkl"); Pte = pd.read_pickle(C / "points_student_test.pkl")
Ftr = Ftr[Ftr.reliable & (Ftr.pt % config.TRAIN_POINT_STRIDE == 0)].reset_index(drop=True)
Fte = Fte[Fte.reliable].reset_index(drop=True)
Pte_r = Pte[Pte.reliable]
arcs_te = {t: np.loadtxt(config.STUDENT_TEST_DIR / t / f"{t}.arcs", dtype=np.int64) for t in list_trips(config.STUDENT_TEST_DIR)}
print("train rows", len(Ftr), "pos rate", Ftr.y.mean().round(3), "| student-test rows", len(Fte))

X, y, groups = Ftr[FEATURES].values, Ftr.y.values, Ftr.trip.values
Xte, yte = Fte[FEATURES].values, Fte.y.values
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups))
(config.ARTIFACT_DIR / "models").mkdir(parents=True, exist_ok=True)
(config.RESULTS_DIR).mkdir(exist_ok=True); config.PLOTS_DIR.mkdir(exist_ok=True)


def cand_metrics(yt, p, df):
    pred = (p >= 0.5).astype(int)
    d = df[["trip", "pt", "arc"]].copy(); d["prob"] = p
    ch = top1_choice(d)
    pts = df.drop_duplicates(["trip", "pt"])[["trip", "pt"]].merge(
        pd.concat([Ptr, Pte])[["trip", "pt", "label_arc"]], on=["trip", "pt"])
    return {"macro_f1": f1_score(yt, pred, average="macro"),
            "precision": precision_score(yt, pred, zero_division=0),
            "recall": recall_score(yt, pred, zero_division=0),
            "log_loss": log_loss(yt, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
            "top1": observation_top1(pts, ch)}, ch


rows, fold_rows = [], []
for name in args.models.split(","):
    family, cfg, make = MODEL_SPECS[name]
    print(f"\n== {name}")
    fm = []
    for k, (tr, va) in enumerate(folds):
        m = make(); t0 = time.time(); m.fit(X[tr], y[tr]); ft = time.time() - t0
        p = m.predict_proba(X[va])[:, 1]
        res, _ = cand_metrics(y[va], p, Ftr.iloc[va])
        res.update(model=name, fold=k, fit_s=ft); fm.append(res); fold_rows.append(res)
        print(f"  fold {k}: macroF1={res['macro_f1']:.4f} top1={res['top1']:.4f} ll={res['log_loss']:.4f} ({ft:.0f}s)")
    fm = pd.DataFrame(fm)
    # refit on all training rows, evaluate on student test
    m = make(); t0 = time.time(); m.fit(X, y); train_s = time.time() - t0
    t0 = time.time(); pte = m.predict_proba(Xte)[:, 1]; inf_s = time.time() - t0
    te, ch = cand_metrics(yte, pte, Fte)
    seq = Pte_r.merge(ch, on=["trip", "pt"], how="left")
    cont = np.mean([route_continuity(g.pred_arc.fillna(-1).astype(int).values, arcs_te[t]) for t, g in seq.groupby("trip")])
    miss = 1 - Pte_r.label_in.mean()
    cm = confusion_matrix(yte, (pte >= .5).astype(int))
    joblib.dump(m, config.ARTIFACT_DIR / "models" / f"{name}.joblib", compress=3)
    np.save(config.CACHE_DIR / f"pte_{name}.npy", pte)
    extra = {}
    if hasattr(m, "steps") and hasattr(m.steps[-1][1], "loss_curve_"):
        mlp = m.steps[-1][1]; extra = {"loss_curve": mlp.loss_curve_, "val_scores": mlp.validation_scores_}
        json.dump(extra, open(config.RESULTS_DIR / f"history_{name}.json", "w"))
    if hasattr(m, "validation_score_") and len(getattr(m, "validation_score_", [])):
        json.dump({"train_score": list(m.train_score_), "val_score": list(m.validation_score_)},
                  open(config.RESULTS_DIR / f"history_{name}.json", "w"))
    rows.append({"model": name, "family": family, "config": cfg,
                 "cv_macro_f1_mean": fm.macro_f1.mean(), "cv_macro_f1_std": fm.macro_f1.std(),
                 "cv_top1_mean": fm.top1.mean(), "cv_log_loss_mean": fm.log_loss.mean(),
                 "test_top1": te["top1"], "test_macro_f1": te["macro_f1"], "test_precision": te["precision"],
                 "test_recall": te["recall"], "test_log_loss": te["log_loss"],
                 "test_route_continuity_clf_only": cont, "candidate_miss_rate": miss,
                 "train_time_s": train_s, "inference_ms_per_1k_points": inf_s / Pte_r.shape[0] * 1e6,
                 "train_macro_f1": f1_score(y, (m.predict_proba(X)[:, 1] >= .5).astype(int), average="macro"),
                 "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]})
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()})

res = pd.DataFrame(rows)
out = config.RESULTS_DIR / "model_comparison.csv"
if out.exists() and args.models != ",".join(MODEL_SPECS):
    old = pd.read_csv(out); res = pd.concat([old[~old.model.isin(res.model)], res])
res.to_csv(out, index=False)
fr = pd.DataFrame(fold_rows); fo = config.RESULTS_DIR / "cv_folds.csv"
if fo.exists() and args.models != ",".join(MODEL_SPECS):
    old = pd.read_csv(fo); fr = pd.concat([old[~old.model.isin(fr.model)], fr])
fr.to_csv(fo, index=False)
