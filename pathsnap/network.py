"""Per-trip road network: projected geometry, spatial index and directed graph."""
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp

from .geo import LocalProjection, bearing_xy
from . import config


class RoadNetwork:
    def __init__(self, nodes, arcs, proj):
        self.proj = proj
        self.nxy = proj.to_xy(nodes[:, 0], nodes[:, 1])
        self.arcs = arcs
        self.n_nodes = len(nodes)
        a, b = self.nxy[arcs[:, 0]], self.nxy[arcs[:, 1]]
        self.ax, self.ay, self.bx, self.by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_xy(self.bx - self.ax, self.by - self.ay)
        self.out_deg = np.bincount(arcs[:, 0], minlength=self.n_nodes)
        self.in_deg = np.bincount(arcs[:, 1], minlength=self.n_nodes)
        pair = arcs[:, 0].astype(np.int64) * (self.n_nodes + 1) + arcs[:, 1]
        rev = arcs[:, 1].astype(np.int64) * (self.n_nodes + 1) + arcs[:, 0]
        self.has_twin = np.isin(rev, pair)
        self._build_index()
        self._graph = None

    def _build_index(self):
        step = config.DENSIFY_STEP_M
        n = np.maximum(1, np.ceil(self.length / step).astype(int)) + 1
        arc_id = np.repeat(np.arange(len(self.arcs)), n)
        offs = np.concatenate([np.linspace(0, 1, k) for k in n]) if len(n) else np.zeros(0)
        xs = self.ax[arc_id] + offs * (self.bx - self.ax)[arc_id]
        ys = self.ay[arc_id] + offs * (self.by - self.ay)[arc_id]
        self.sample_arc = arc_id
        self.tree = cKDTree(np.column_stack([xs, ys]))
        self.index_slack = step / 2 + 1e-6

    @property
    def csr(self):
        if self._graph is None:
            n = self.n_nodes
            # keep shortest arc if parallel duplicates exist
            g = sp.coo_matrix((self.length + 1e-3, (self.arcs[:, 0], self.arcs[:, 1])), shape=(n, n)).tocsr()
            self._graph = g
        return self._graph

    def arc_lookup(self):
        """dict (from,to)->arc_id (shortest if duplicated)."""
        if not hasattr(self, "_lookup"):
            order = np.argsort(-self.length)
            self._lookup = {(int(self.arcs[i, 0]), int(self.arcs[i, 1])): int(i) for i in order}
        return self._lookup


def build_trip(trip):
    """Attach projected track + RoadNetwork to a loaded trip dict."""
    nodes = trip["nodes"]
    proj = LocalProjection(nodes[:, 0].mean(), nodes[:, 1].mean())
    net = RoadNetwork(nodes, trip["arcs"], proj)
    tr = trip["track"]
    xy = proj.to_xy(tr[:, 0], tr[:, 1])
    trip.update({"proj": proj, "net": net, "xy": xy, "t": tr[:, 2]})
    return trip
