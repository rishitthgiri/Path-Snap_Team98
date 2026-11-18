"""Candidate generation: arcs within a radius of each GPS point (top-K by distance)."""
import numpy as np
from .geo import point_segment
from . import config


def generate_candidates(xy, net, radius=None, max_k=None):
    """Returns flat arrays (point_idx, arc_id, dist, frac, frac_raw), sorted by
    point then distance."""
    radius = radius or config.CAND_RADIUS_M
    max_k = max_k or config.CAND_MAX_K
    hits = net.tree.query_ball_point(xy, radius + net.index_slack)
    pi, ai = [], []
    for t, h in enumerate(hits):
        if h:
            arcs = np.unique(net.sample_arc[h])
            pi.append(np.full(len(arcs), t))
            ai.append(arcs)
    if not pi:
        z = np.zeros(0)
        return z.astype(int), z.astype(int), z, z, z
    pi, ai = np.concatenate(pi), np.concatenate(ai)
    d, u, uraw = point_segment(xy[pi, 0], xy[pi, 1], net.ax[ai], net.ay[ai], net.bx[ai], net.by[ai])
    keep = d <= radius
    pi, ai, d, u, uraw = pi[keep], ai[keep], d[keep], u[keep], uraw[keep]
    order = np.lexsort((d, pi))
    pi, ai, d, u, uraw = pi[order], ai[order], d[order], u[order], uraw[order]
    # rank within point, keep top-K
    start = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
    rank = np.arange(len(pi)) - np.repeat(start, np.diff(np.r_[start, len(pi)]))
    keep = rank < max_k
    return pi[keep], ai[keep], d[keep], u[keep], uraw[keep]
