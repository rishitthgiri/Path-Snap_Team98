"""Run the full inference pipeline for trips from the command line (same code path as the API).

Usage: python scripts/06_predict_cli.py --split student_test [--trip 00000007]
"""
import _common  # noqa
import argparse
import pandas as pd
from pathsnap import config
from pathsnap.data_io import list_trips
from pathsnap.inference import Predictor, evaluate_prediction

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="student_test"); ap.add_argument("--trip")
a = ap.parse_args()
sdir = config.DATA_DIR / a.split
P = Predictor(); rows = []
for tid in ([a.trip] if a.trip else list_trips(sdir)):
    pred = P.predict_trip(sdir, tid); m = evaluate_prediction(sdir, tid, pred)
    rows.append({"trip": tid, **m, "seconds": pred["timing_s"]["total"]})
    print(tid, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)})
df = pd.DataFrame(rows); df.to_csv(config.RESULTS_DIR / f"pipeline_{a.split}.csv", index=False)
print(df.mean(numeric_only=True).round(4).to_string())
