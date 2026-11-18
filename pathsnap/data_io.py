"""Reading raw trip files (.track, .nodes, .arcs, .route)."""
from pathlib import Path
import numpy as np


def list_trips(split_dir):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def load_trip(split_dir, trip_id, with_route=True):
    d = Path(split_dir) / trip_id
    track = np.loadtxt(d / f"{trip_id}.track", ndmin=2)       # lon, lat, t
    nodes = np.loadtxt(d / f"{trip_id}.nodes", ndmin=2)       # lon, lat
    arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)  # from, to
    route = None
    rp = d / f"{trip_id}.route"
    if with_route and rp.exists():
        route = np.loadtxt(rp, dtype=np.int64, ndmin=1).ravel()
    return {"trip_id": trip_id, "track": track, "nodes": nodes,
            "arcs": arcs, "route": route}


def clean_track(track):
    """Drop rows with NaN/implausible coords and non-increasing timestamps.
    Returns cleaned track and the index of kept rows (into the raw file)."""
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    idx = np.flatnonzero(ok)
    tr = track[idx]
    # keep strictly increasing timestamps
    keep = np.ones(len(tr), bool)
    last = -np.inf
    for i, t in enumerate(tr[:, 2]):
        if t <= last:
            keep[i] = False
        else:
            last = t
    return tr[keep], idx[keep]
