"""HMM + Viterbi route reconstruction.

Hidden states   : for GPS point t, the candidate arcs C_t (same candidate set the
                  classifier scored, so state k <-> arc id is preserved).
Emission        : log P(obs_t | arc) = w_e * log(classifier probability of arc).
Transition      : Newson & Krumm (2009) style. For arcs a (at t-1) and b (at t)
                  d_route = network driving distance from the projection on a to the
                  projection on b (same arc: forward offset; different arc:
                  rest of a + shortest path(to_a -> from_b) + offset on b),
                  d_gps   = straight-line distance between the two GPS points.
                  log P(b | a) = -|d_route - d_gps| / (beta * sqrt(max(dt,1))).
                  Moving backwards on a one-direction arc more than `back_tol` metres
                  or unreachable within the search limit gets `unreachable_logp`
                  (a finite penalty, so decoding never breaks on disconnected pairs).
GPS gaps        : the shortest-path search limit and beta grow with the time gap.
Missing cands   : points with no candidate are skipped and returned as -1; the chain
                  continues from the last point that had candidates.
Decoding        : standard Viterbi in log space,
                  V_t(b) = e_t(b) + max_a [V_{t-1}(a) + log P(b|a)].
"""
import numpy as np
from scipy.sparse.csgraph import dijkstra
from . import config


class PathCache:
    def __init__(self, net):
        self.net, self.g, self.cache = net, net.csr, {}

    def dists_from(self, node, limit):
        c = self.cache.get(node)
        if c is None or c[0] < limit:
            d = dijkstra(self.g, directed=True, indices=node, limit=limit)
            nz = np.flatnonzero(np.isfinite(d))
            c = (limit, dict(zip(nz.tolist(), d[nz].tolist())))
            self.cache[node] = c
        return c[1]


def viterbi_decode(trip, cand, params=None):
    """cand: DataFrame with columns pt, arc, prob, (dist), proj_frac for ONE trip.
    Returns array (len track) of decoded arc ids (-1 where no candidates)."""
    p = dict(config.HMM_DEFAULTS); p.update(params or {})
    net, xy, t = trip["net"], trip["xy"], trip["t"]
    T = len(xy)
    cand = cand.sort_values(["pt", "prob"], ascending=[True, False])
    pts = cand.pt.values
    starts = np.r_[0, np.flatnonzero(np.diff(pts)) + 1, len(pts)]
    arcs_all, frac_all = cand.arc.values.astype(np.int64), cand.proj_frac.values.astype(float)
    logp_all = p["emission_weight"] * np.log(np.clip(cand.prob.values.astype(float), 1e-6, 1.0))
    cache = PathCache(net)
    L, A = net.length, net.arcs

    step_pts, step_arcs, back = [], [], []
    V = None; prev = None
    for s in range(len(starts) - 1):
        i0, i1 = starts[s], starts[s + 1]
        tp = int(pts[i0]); arcs = arcs_all[i0:i1]; pos = frac_all[i0:i1] * L[arcs]; e = logp_all[i0:i1]
        if V is None:
            V = e.copy(); bp = np.full(len(arcs), -1)
        else:
            ptp, parcs, ppos = prev
            dt = max(float(t[tp] - t[ptp]), 1.0)
            dgps = float(np.hypot(*(xy[tp] - xy[ptp])))
            limit = min(100.0 + 40.0 * dt + dgps, 8000.0)
            beta = p["beta"] * np.sqrt(dt)
            trans = np.full((len(parcs), len(arcs)), p["unreachable_logp"])
            for i, a in enumerate(parcs):
                dmap = cache.dists_from(int(A[a, 1]), limit)
                rem = L[a] - ppos[i]
                for j, b in enumerate(arcs):
                    if a == b:
                        dr = pos[j] - ppos[i]
                        if dr < -p["back_tol"]:
                            continue
                        dr = max(dr, 0.0)
                    else:
                        sp = dmap.get(int(A[b, 0]))
                        if sp is None:
                            continue
                        dr = rem + sp + pos[j]
                    trans[i, j] = -abs(dr - dgps) / beta
            tot = V[:, None] + trans
            bp = tot.argmax(0)
            V = tot[bp, np.arange(len(arcs))] + e
            V -= V.max()
        step_pts.append(tp); step_arcs.append(arcs); back.append(bp)
        prev = (tp, arcs, pos)

    out = np.full(T, -1, dtype=np.int64)
    if V is None:
        return out
    k = int(np.argmax(V))
    for s in range(len(step_pts) - 1, -1, -1):
        out[step_pts[s]] = step_arcs[s][k]
        k = back[s][k]
    return out


def reconstruct_route(decoded, net):
    """Collapse per-point arcs into an arc sequence and bridge non-adjacent pairs
    with the network shortest path. Returns (route arc list, n_bridged, n_unbridged)."""
    seq = []
    for a in decoded:
        if a >= 0 and (not seq or seq[-1] != a):
            seq.append(int(a))
    if not seq:
        return [], 0, 0
    lookup = net.arc_lookup()
    route, bridged, broken = [seq[0]], 0, 0
    for a, b in zip(seq[:-1], seq[1:]):
        u, v = int(net.arcs[a, 1]), int(net.arcs[b, 0])
        if u != v:
            d, pred = dijkstra(net.csr, directed=True, indices=u, return_predecessors=True, limit=20000)
            if np.isfinite(d[v]):
                path, x = [], v
                while x != u:
                    path.append(x); x = pred[x]
                path.append(u); path = path[::-1]
                route.extend(lookup[(path[i], path[i + 1])] for i in range(len(path) - 1))
                bridged += 1
            else:
                broken += 1
        route.append(b)
    return route, bridged, broken
