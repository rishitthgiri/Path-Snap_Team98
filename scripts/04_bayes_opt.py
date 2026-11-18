"""Bayesian hyperparameter optimisation of the selected classifier (HistGradientBoosting).

Objective : maximise mean observation-level Top-1 accuracy over GroupKFold(3) on the
            80 training trips (every 10th reliable point to keep each trial cheap).
Search    : learning_rate (log 0.02-0.3), max_leaf_nodes (int 15-127),
            min_samples_leaf (int 10-300), l2_regularization (log 1e-4-10),
            max_features (0.4-1.0). max_iter=500 with early stopping.
Trials    : 6 random + 19 GP-EI = 25.
Then      : refit best config on all training rows (stride 5, as in 03), evaluate on
            the 10 student-test trips and compare with the untuned model.
Usage     : python scripts/04_bayes_opt.py
"""
import _common  # noqa
import json, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.metrics import top1_choice, observation_top1
from pathsnap.bayesopt import Space, bayes_opt

C = config.CACHE_DIR
Ftr = pd.read_pickle(C / "features_train.pkl"); Ptr = pd.read_pickle(C / "points_train.pkl")
Fte = pd.read_pickle(C / "features_student_test.pkl"); Pte = pd.read_pickle(C / "points_student_test.pkl")
Ftr = Ftr[Ftr.reliable].reset_index(drop=True); Fte = Fte[Fte.reliable].reset_index(drop=True)
Fbo = Ftr[Ftr.pt % 10 == 0].reset_index(drop=True)
Ffull = Ftr[Ftr.pt % config.TRAIN_POINT_STRIDE == 0].reset_index(drop=True)
LAB = pd.concat([Ptr, Pte])[["trip", "pt", "label_arc"]]


def evaluate(m, df):
    p = m.predict_proba(df[FEATURES].values)[:, 1]
    d = df[["trip", "pt", "arc"]].copy(); d["prob"] = p
    pts = df.drop_duplicates(["trip", "pt"])[["trip", "pt"]].merge(LAB, on=["trip", "pt"])
    return {"top1": observation_top1(pts, top1_choice(d)),
            "macro_f1": f1_score(df.y, (p >= .5).astype(int), average="macro"),
            "log_loss": log_loss(df.y, np.clip(p, 1e-7, 1 - 1e-7))}


def make(params):
    return HistGradientBoostingClassifier(max_iter=500, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=config.SEED, **params)


folds = list(GroupKFold(n_splits=3).split(Fbo, Fbo.y, Fbo.trip))


def objective(params):
    t0 = time.time(); r = []
    for tr, va in folds:
        m = make(params).fit(Fbo.iloc[tr][FEATURES].values, Fbo.y.values[tr]); r.append(evaluate(m, Fbo.iloc[va]))
    r = pd.DataFrame(r)
    return r.top1.mean(), {"cv_macro_f1": r.macro_f1.mean(), "cv_log_loss": r.log_loss.mean(),
                           "cv_top1_std": r.top1.std(), "seconds": time.time() - t0}


space = Space([("learning_rate", 0.02, 0.3, "log"), ("max_leaf_nodes", 15, 127, "int"),
               ("min_samples_leaf", 10, 300, "int"), ("l2_regularization", 1e-4, 10.0, "log"),
               ("max_features", 0.4, 1.0, "float")])
t0 = time.time()
trials = bayes_opt(objective, space, n_init=6, n_iter=19, seed=config.SEED)
bo_time = time.time() - t0
T = pd.DataFrame(trials); T.to_csv(config.RESULTS_DIR / "bayes_opt_trials.csv", index=False)
best = T.loc[T.objective.idxmax()]
best_params = {k: (int(best[k]) if k in ("max_leaf_nodes", "min_samples_leaf") else float(best[k]))
               for k in ["learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_features"]}
print("best", best_params, best.objective)

# untuned vs tuned, same data + same metrics
untuned = dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0, max_features=1.0)
rows = []
for name, prm in [("HGB untuned", untuned), ("HGB tuned (BO)", best_params)]:
    cv = []
    for tr, va in GroupKFold(n_splits=5).split(Ffull, Ffull.y, Ffull.trip):
        m = make(prm).fit(Ffull.iloc[tr][FEATURES].values, Ffull.y.values[tr]); cv.append(evaluate(m, Ffull.iloc[va]))
    cv = pd.DataFrame(cv)
    t1 = time.time(); m = make(prm).fit(Ffull[FEATURES].values, Ffull.y.values); fit_s = time.time() - t1
    t1 = time.time(); te = evaluate(m, Fte); inf = time.time() - t1
    rows.append({"model": name, **prm, "cv_macro_f1": cv.macro_f1.mean(), "cv_top1": cv.top1.mean(),
                 "cv_log_loss": cv.log_loss.mean(), "test_top1": te["top1"], "test_macro_f1": te["macro_f1"],
                 "test_log_loss": te["log_loss"], "fit_s": fit_s, "n_iter_used": m.n_iter_,
                 "inference_ms_per_1k_points": inf / Pte.reliable.sum() * 1e6})
    if "tuned (BO)" in name:
        joblib.dump(m, config.ARTIFACT_DIR / "classifier.joblib", compress=3)
cmp = pd.DataFrame(rows); cmp.to_csv(config.RESULTS_DIR / "tuned_vs_untuned.csv", index=False)
json.dump({"best_params": best_params, "n_trials": len(T), "bo_total_seconds": bo_time,
           "objective": "mean GroupKFold(3) observation-level Top-1 on train trips"},
          open(config.RESULTS_DIR / "bayes_opt_summary.json", "w"), indent=2)
print(cmp.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(T.trial, T.objective, "o", label="trial"); ax[0].plot(T.trial, T.objective.cummax(), "-", label="best so far")
ax[0].axvline(5.5, ls="--", c="grey"); ax[0].text(0.3, T.objective.min(), "random init", fontsize=8)
ax[0].set(title="Bayesian optimisation (GP-EI)", xlabel="trial", ylabel="CV Top-1"); ax[0].legend()
mt = ["cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1"]
x = np.arange(len(mt)); ax[1].bar(x - .2, cmp.iloc[0][mt], .4, label="untuned"); ax[1].bar(x + .2, cmp.iloc[1][mt], .4, label="tuned")
ax[1].set_xticks(x, mt); ax[1].set_ylim(cmp[mt].values.min() - .02, 1); ax[1].legend(); ax[1].set_title("Untuned vs tuned HGB")
plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "bayes_opt.png", dpi=120); plt.close()
