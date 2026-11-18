"""Folium maps: raw GPS, nearby road network, reference route and predicted route.

Usage: python scripts/07_make_maps.py [--split student_test] [--trips 00000009,00000052]
Writes maps/pred_<split>_<trip>.html (open in a browser; screenshot for the report).
"""
import _common  # noqa
import argparse
from pathsnap import config
from pathsnap.data_io import list_trips, load_trip
from pathsnap.inference import Predictor, evaluate_prediction
from pathsnap.maps import trip_map

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="student_test"); ap.add_argument("--trips")
a = ap.parse_args()
sdir = config.DATA_DIR / a.split
config.MAPS_DIR.mkdir(exist_ok=True)
P = Predictor()
for tid in (a.trips.split(",") if a.trips else list_trips(sdir)):
    pred = P.predict_trip(sdir, tid); m = evaluate_prediction(sdir, tid, pred)
    trip = load_trip(sdir, tid)
    title = f"{a.split} {tid}: point acc {m['hmm_point_accuracy']:.3f}, route F1 {m['route_f1']:.3f}"
    trip_map(trip, pred_route=pred["route"], title=title).save(str(config.MAPS_DIR / f"pred_{a.split}_{tid}.html"))
    print("wrote", tid)
