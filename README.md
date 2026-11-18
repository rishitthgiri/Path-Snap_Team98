# PathSnap – map-matching noisy GPS to a road network

Classifier scores candidate road segments for every GPS point → HMM/Viterbi picks a
connected route → FastAPI serves it → browser UI shows it.

## Pipeline

```
.track/.nodes/.arcs ─► clean + local metric projection ─► candidate arcs (≤12 within 50 m)
      ─► 20 features per (point, arc) ─► HistGradientBoosting P(arc is correct)
      ─► HMM / Viterbi over candidates (network shortest paths) ─► route (gaps bridged)
.route (server side only) ─► per-point labels ─► accuracy / route F1 / continuity
```

| Step | Where | Notes |
|---|---|---|
| Read + clean | `pathsnap/data_io.py` | drops NaN / out-of-range / non-increasing timestamps |
| Projection | `pathsnap/geo.py` | per-trip equirectangular metres (error ≪ GPS noise at city scale) |
| Road network | `pathsnap/network.py` | arc geometry, twin (two-way) flags, degrees, KD-tree on 10 m arc samples, sparse graph |
| Labels | `pathsnap/labels.py` | `.route` gives arc order only → monotonic DP aligns each GPS point to a route arc. Points > 50 m from the route are marked unreliable and excluded from training/scoring |
| Candidates | `pathsnap/candidates.py` | radius 50 m, top-12 by distance |
| Features | `pathsnap/features.py` | distance + rank, heading difference, along-arc motion (direction of travel), speed, context distance of ±5 neighbours, projection position, arc length, twin flag, degrees, time gaps |
| Models | `pathsnap/models.py` | 8 scikit-learn models from 6 families |
| Bayesian optimisation | `pathsnap/bayesopt.py` | own GP + Expected-Improvement (scikit-learn GP, Matern 5/2) |
| HMM / Viterbi | `pathsnap/hmm.py` | emissions = classifier log-prob; transitions = Newson–Krumm route-vs-GPS distance |
| Inference | `pathsnap/inference.py` | shared by API and CLI; reads `.route` only after prediction |
| API | `api/main.py` | FastAPI: `/health`, `/trips`, `/predict` |
| UI | `ui/index.html`, `ui/serve_ui.py` | Leaflet map + metrics; talks to the API only |
| Maps | `pathsnap/maps.py`, `scripts/07_make_maps.py` | Folium: GPS, network, reference, prediction, layer control, legend |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Put the data at `data/train/<trip>/…` and `data/student_test/<trip>/…`, or set `PATHSNAP_DATA=/path/to/folder`.

## Reproduce everything (in order)

```bash
python scripts/01_explore_data.py        # EDA tables, plots, Folium EDA maps
python scripts/02_build_features.py      # candidates + features + labels -> cache/
python scripts/03_train_compare.py       # 8 models, GroupKFold(5) CV + student-test  (~25 min on 2 cores)
python scripts/03b_plot_comparison.py    # comparison plots
python scripts/04_bayes_opt.py           # Bayesian optimisation of HGB, saves artifacts/classifier.joblib (~8 min)
python scripts/05_hmm_eval.py            # tune HMM, classifier-only vs +HMM, saves artifacts/inference_config.json
python scripts/06_predict_cli.py         # full pipeline on student_test (same code as API)
python scripts/07_make_maps.py --trips 00000009,00000077   # Folium prediction maps -> maps/
python scripts/08_static_pred_plots.py   # PNG prediction maps for the report
```
Seed: 42 everywhere. All metrics in `results/`, all figures in `plots/`.

## Evaluation day (two laptops)

**Laptop 1 – API** (copy the repo incl. `artifacts/`; put the 10 evaluator trips in a folder):
```bash
# Windows PowerShell: $env:PATHSNAP_EVAL_DATA="C:\path\to\evaluator_test"
export PATHSNAP_EVAL_DATA=/path/to/evaluator_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
The terminal shows a `REQUEST` and a `RESULT` log line per prediction (show these to the evaluator).
If `PATHSNAP_EVAL_DATA` is not set or doesn't exist, the API serves `data/student_test` (for rehearsal).
Find Laptop 1's IP with `ipconfig` / `ip a`; allow port 8000 through the firewall.

**Laptop 2 – UI** (needs only Python and a browser):
```bash
python ui/serve_ui.py --port 8501
# open http://localhost:8501/?api=http://<LAPTOP1_IP>:8000
```
Click **Load evaluation trips** → pick a trip → **Predict route**. Map tiles + Leaflet come from a CDN, so Laptop 2 needs internet.
Ground truth never leaves Laptop 1: the API returns GPS points, the predicted route and metrics only.

## Results (student-test = 10 trips; CV = GroupKFold(5) over the 80 train trips)

Primary selection metric: **observation-level Top-1 accuracy** (argmax candidate per GPS point vs. reference arc). It is
what the decoder consumes and what the evaluator sees. Log loss is the second criterion, because the HMM uses the probabilities.

| Model | Family | CV Macro F1 | Test Top-1 (clf) | Test Top-1 (+HMM) | Test Macro F1 | Test log loss | Continuity clf → HMM | Inference ms / 1k pts | Train Macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| HistGradientBoosting | Boosting | 0.9332 ± 0.0227 | 0.8890 | 0.9358 | 0.9329 | 0.0751 | 0.764 → 0.912 | 14.2 | 0.9475 |
| RandomForest | Ensemble (bagging) | 0.9330 ± 0.0225 | 0.8860 | 0.9402 | 0.9328 | 0.0743 | 0.760 → 0.912 | 48.8 | 0.9688 |
| MLP | Neural network | 0.9304 ± 0.0217 | 0.8808 | 0.9158 | 0.9273 | 0.0814 | 0.760 → 0.910 | 1.7 | 0.9438 |
| DecisionTree | Tree-based | 0.9275 ± 0.0225 | 0.8803 | 0.9286 | 0.9251 | 0.1112 | 0.755 → 0.907 | 0.4 | 0.9449 |
| ExtraTrees | Ensemble (bagging) | 0.9294 ± 0.0210 | 0.8792 | 0.9366 | 0.9272 | 0.0781 | 0.749 → 0.912 | 67.2 | 0.9481 |
| LogisticRegression | Linear / probabilistic | 0.9211 ± 0.0204 | 0.8773 | 0.9210 | 0.9178 | 0.0947 | 0.756 → 0.905 | 0.3 | 0.9220 |
| KNN | Instance-based | 0.9198 ± 0.0223 | 0.8673 | 0.9002 | 0.9172 | 0.1236 | 0.733 → 0.908 | 3105.9 | 1.0000 |
| GaussianNB | Linear / probabilistic | 0.8572 ± 0.0169 | 0.8476 | 0.9024 | 0.8436 | 0.5995 | 0.712 → 0.889 | 0.9 | 0.8561 |

Bayesian optimisation (25 trials, 6 random + 19 GP-EI, objective = GroupKFold(3) Top-1):

| Model | CV Macro F1 | CV Top-1 | Test Top-1 | Test Macro F1 | Test log loss | Inference ms / 1k pts |
|---|---|---|---|---|---|---|
| HGB untuned | 0.9324 | 0.8901 | 0.8901 | 0.9333 | 0.0741 | 16.9 |
| HGB tuned (BO) | 0.9339 | 0.8915 | 0.8951 | 0.9362 | 0.0703 | 83.4 |

Final system (tuned HGB + HMM, student test): Top-1 **0.895 → 0.936**, route continuity
**0.768 → 0.913** (reference labels: 0.919), length-weighted route F1
**0.942 → 0.986** (precision 0.976, recall 0.996).
About 0.6 s per trip end-to-end.

## Known limitations / assumptions
- Per-point labels are **derived** from `.route` by alignment; they are not observed. ~11 % of train points are > 50 m from the reference route
  (74 % of them from trip 00000054, whose GPS runs well beyond its route) and are excluded. Two train trips have
  multi-hour time gaps (00000078, 00000090); the HMM widens its search limit with the gap.
- Route continuity counts only directly-adjacent arc pairs. At 1 Hz a vehicle can skip a very short arc between two
  fixes, so even the reference labels score ~0.92; the final route bridges such gaps by shortest path (≈1.0).
- Remaining errors are mostly twin arcs (same road, opposite direction) when the vehicle is slow/stationary, and parallel roads.
- Evaluator data is never used for training, tuning or thresholds.
