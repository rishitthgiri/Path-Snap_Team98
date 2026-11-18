"""Load + clean + project a trip, derive labels and build candidate features."""
import numpy as np
from .data_io import load_trip, clean_track
from .network import build_trip
from .labels import align_points_to_route
from .features import build_features


def prepare_trip(split_dir, trip_id, with_labels=True, point_stride=1):
    trip = load_trip(split_dir, trip_id, with_route=with_labels)
    trip["raw_n"] = len(trip["track"])
    trip["track"], trip["kept_idx"] = clean_track(trip["track"])
    build_trip(trip)
    label_arc = None
    if with_labels and trip["route"] is not None:
        label_arc, label_dist, _ = align_points_to_route(trip["xy"], trip["net"], trip["route"])
        trip["label_arc"], trip["label_dist"] = label_arc, label_dist
    trip["features"] = build_features(trip, label_arc, point_stride)
    return trip
