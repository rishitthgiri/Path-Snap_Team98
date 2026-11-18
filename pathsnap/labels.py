"""Derive a per-observation reference arc from the trip-level reference route.

The .route file lists the arcs travelled in order, but not which GPS point lies on
which arc. We align points to route arcs with a monotonic dynamic programme:
point t may only be assigned to route position i >= the position of point t-1,
and the total point-to-arc distance is minimised. This respects the order of travel
(handles loops / revisits) unlike simple nearest-route-arc assignment.

Limitation: labels are derived, not observed. Points far from every route arc
(GPS outliers, route errors) still get a label; `label_dist` records how far.
"""
import numpy as np
from .geo import point_segment


def align_points_to_route(xy, net, route):
    T, M = len(xy), len(route)
    px, py = xy[:, 0][:, None], xy[:, 1][:, None]
    d, _, _ = point_segment(px, py, net.ax[route][None], net.ay[route][None],
                            net.bx[route][None], net.by[route][None])  # T x M
    D = np.empty((T, M))
    back = np.empty((T, M), dtype=np.int32)
    D[0] = d[0]
    for t in range(1, T):
        prev = D[t - 1]
        # prefix minimum and its argmin over j <= i
        pm = np.minimum.accumulate(prev)
        idx = np.arange(M)
        is_new = np.r_[True, prev[1:] < pm[:-1]]
        arg = np.maximum.accumulate(np.where(is_new, idx, 0))
        D[t] = d[t] + pm
        back[t] = arg
    pos = np.empty(T, dtype=np.int64)
    pos[-1] = int(np.argmin(D[-1]))
    for t in range(T - 1, 0, -1):
        pos[t - 1] = back[t, pos[t]]
    label_arc = route[pos]
    label_dist = d[np.arange(T), pos]
    return label_arc, label_dist, pos
