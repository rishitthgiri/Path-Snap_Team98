"""Plots for the model comparison (all from results/*.csv produced by 03_train_compare.py)."""
import _common  # noqa
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathsnap import config

R, P = config.RESULTS_DIR, config.PLOTS_DIR
res = pd.read_csv(R / "model_comparison.csv").sort_values("test_top1")
folds = pd.read_csv(R / "cv_folds.csv")

fig, ax = plt.subplots(figsize=(9, 4.5))
order = res.model.tolist()
ax.boxplot([folds[folds.model == m].macro_f1 for m in order], tick_labels=order, vert=False)
ax.set(title="Cross-validation Macro F1 (GroupKFold=5 over 80 train trips)", xlabel="Macro F1")
plt.tight_layout(); plt.savefig(P / "cv_macro_f1.png", dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(10, 4.5)); x = np.arange(len(res)); w = .38
ax.barh(x - w / 2, res.test_top1, w, label="Top-1 accuracy"); ax.barh(x + w / 2, res.test_macro_f1, w, label="Macro F1")
ax.set_yticks(x, res.model); ax.set_xlim(min(res.test_macro_f1.min(), res.test_top1.min()) - .05, 1)
ax.set(title="Student-test performance (10 trips)"); ax.legend(loc="lower right")
for i, (a, b) in enumerate(zip(res.test_top1, res.test_macro_f1)):
    ax.text(a, i - w / 2, f" {a:.3f}", va="center", fontsize=8); ax.text(b, i + w / 2, f" {b:.3f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(P / "student_test_top1_f1.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].barh(res.model, res.test_log_loss); ax[0].set(title="Student-test log loss (lower is better)", xscale="log")
ax[1].barh(res.model, res.inference_ms_per_1k_points); ax[1].set(title="Inference time (ms per 1000 GPS points)", xscale="log")
plt.tight_layout(); plt.savefig(P / "logloss_inference.png", dpi=120); plt.close()

fig, axes = plt.subplots(2, (len(res) + 1) // 2, figsize=(4 * ((len(res) + 1) // 2), 7))
for a, (_, r) in zip(axes.ravel(), res.iterrows()):
    ConfusionMatrixDisplay(np.array([[r.tn, r.fp], [r.fn, r.tp]]), display_labels=["other", "true arc"]).plot(ax=a, colorbar=False, values_format="d")
    a.set_title(r.model, fontsize=10)
for a in axes.ravel()[len(res):]: a.axis("off")
plt.tight_layout(); plt.savefig(P / "confusion_matrices.png", dpi=110); plt.close()

for f in R.glob("history_*.json"):
    h = json.load(open(f)); name = f.stem.replace("history_", "")
    fig, ax = plt.subplots(figsize=(6, 4))
    if "loss_curve" in h:
        ax.plot(h["loss_curve"], label="training loss (log loss)")
        ax2 = ax.twinx(); ax2.plot(h["val_scores"], color="tab:orange", label="validation accuracy"); ax2.set_ylabel("val accuracy")
        ax2.legend(loc="center right")
    else:
        ax.plot(-np.array(h["train_score"]), label="training loss"); ax.plot(-np.array(h["val_score"]), label="validation loss")
    ax.set(title=f"{name}: training history (fit on all 80 train trips)", xlabel="epoch / iteration", ylabel="loss"); ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(P / f"history_{name}.png", dpi=120); plt.close()
print(res[["model", "cv_macro_f1_mean", "cv_top1_mean", "test_top1", "test_macro_f1", "test_log_loss",
           "test_route_continuity_clf_only", "inference_ms_per_1k_points", "train_macro_f1"]].round(4).to_string(index=False))
