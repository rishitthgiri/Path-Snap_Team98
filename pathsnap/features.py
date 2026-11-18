"""Feature engineering for (GPS observation, candidate arc) pairs.

Each row = one candidate arc for one GPS point. The classifier scores how likely
that arc is the reference arc. Features describe geometry (distance, position on
arc), motion agreement (heading, along-arc movement, speed) and local context
(how well neighbouring points fit the same arc) plus simple topology.
"""
import numpy as np
import pandas as pd

from .geo import point_segment, bearing_xy, angle_diff
from .candidates import generate_candidates
from . import config

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min", "n_cand",
    "heading_diff", "heading_valid", "speed", "along_motion", "along_motion_norm",
    "ctx_dist_mean", "ctx_dist_max", "proj_frac", "outside_seg",
    "arc_len", "has_twin", "from_in_deg", "to_out_deg", "dt_prev", "dt_next",
]


def track_kinematics(xy, t, half=None):
    half = half or config.HEADING_HALF_WINDOW
    T = len(xy)
    i0 = np.clip(np.arange(T) - half, 0, T - 1)
    i1 = np.clip(np.arange(T) + half, 0, T - 1)
    dx, dy = xy[i1, 0] - xy[i0, 0], xy[i1, 1] - xy[i0, 1]
    disp = np.hypot(dx, dy)
    dtw = np.maximum(t[i1] - t[i0], 1e-6)
    heading = bearing_xy(dx, dy)
    speed = disp / dtw
    valid = disp > 3.0           # heading meaningless when (almost) stationary
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]
    return heading, speed, valid, i0, i1, dt_prev, dt_next


def build_features(trip, label_arc=None, point_stride=1):
    xy, t, net = trip["xy"], trip["t"], trip["net"]
    T = len(xy)
    pts = np.arange(0, T, point_stride)
    pi, ai, d, u, uraw = generate_candidates(xy[pts], net)
    pi = pts[pi]  # map back to full-track index
    heading, speed, valid, i0, i1, dt_prev, dt_next = track_kinematics(xy, t)

    n = len(pi)
    df = pd.DataFrame({"trip": trip["trip_id"], "pt": pi, "arc": ai, "dist": d})
    g = df.groupby("pt")["dist"]
    df["dist_rank"] = g.rank(method="first") - 1
    dmin = g.transform("min")
    df["dist_minus_min"] = d - dmin
    df["dist_ratio_min"] = (d + 1.0) / (dmin + 1.0)
    df["n_cand"] = g.transform("size")

    hd = angle_diff(heading[pi], net.bearing[ai])
    df["heading_diff"] = np.where(valid[pi], hd, 90.0)
    df["heading_valid"] = valid[pi].astype(float)
    df["speed"] = speed[pi]

    # signed movement along the arc direction between t-h and t+h
    ux = (net.bx - net.ax)[ai] / np.maximum(net.length[ai], 1e-6)
    uy = (net.by - net.ay)[ai] / np.maximum(net.length[ai], 1e-6)
    mvx = xy[i1[pi], 0] - xy[i0[pi], 0]
    mvy = xy[i1[pi], 1] - xy[i0[pi], 1]
    along = mvx * ux + mvy * uy
    df["along_motion"] = along
    df["along_motion_norm"] = along / np.maximum(np.hypot(mvx, mvy), 1.0)

    # context: distance of neighbouring points to the same arc
    k = config.CONTEXT_HALF_WINDOW
    offs = np.r_[np.arange(-k, 0), np.arange(1, k + 1)]
    nb = np.clip(pi[:, None] + offs[None], 0, T - 1)
    cd, _, _ = point_segment(xy[nb, 0], xy[nb, 1], net.ax[ai][:, None], net.ay[ai][:, None],
                             net.bx[ai][:, None], net.by[ai][:, None])
    df["ctx_dist_mean"] = cd.mean(1)
    df["ctx_dist_max"] = cd.max(1)

    df["proj_frac"] = u
    df["outside_seg"] = np.maximum(0, np.maximum(-uraw, uraw - 1)) * net.length[ai]
    df["arc_len"] = net.length[ai]
    df["has_twin"] = net.has_twin[ai].astype(float)
    df["from_in_deg"] = net.in_deg[net.arcs[ai, 0]]
    df["to_out_deg"] = net.out_deg[net.arcs[ai, 1]]
    df["dt_prev"] = np.minimum(dt_prev[pi], 600)
    df["dt_next"] = np.minimum(dt_next[pi], 600)
    if label_arc is not None:
        df["y"] = (ai == label_arc[pi]).astype(np.int8)
    for c in FEATURES:
        df[c] = df[c].astype(np.float32)
    return df
