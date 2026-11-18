"""Exploratory data analysis: inventory, data quality, network connectivity, plots, Folium maps.

Usage:  python scripts/01_explore_data.py
Outputs: results/eda_inventory.csv, results/eda_quality_log.md, plots/eda_*.png, maps/eda_*.html
Partitions used: train (80) and student_test (10). Evaluator data is never read here.
"""
import _common  # noqa
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from pathsnap import config
from pathsnap.data_io import list_trips, load_trip, clean_track
from pathsnap.network import build_trip
from pathsnap.labels import align_points_to_route
from pathsnap.geo import haversine

for d in (config.RESULTS_DIR, config.PLOTS_DIR, config.MAPS_DIR):
    d.mkdir(exist_ok=True, parents=True)

rows = []
gap_all = {"train": [], "student_test": []}
labeldist_all = {"train": [], "student_test": []}
for split, sdir in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
    for tid in list_trips(sdir):
        tdir = sdir / tid
        files = {ext: (tdir / f"{tid}.{ext}") for ext in ["track", "nodes", "arcs", "route"]}
        trip = load_trip(sdir, tid)
        raw = trip["track"]
        n_nan = int((~np.isfinite(raw)).any(axis=1).sum())
        dt_raw = np.diff(raw[:, 2])
        trip["track"], _ = clean_track(raw)
        build_trip(trip)
        tr, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
        dt = np.diff(tr[:, 2])
        gap_all[split].append(dt)
        step = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
        speed = step / np.maximum(dt, 1e-6)
        n = len(nodes)
        A = sp.coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
        n_weak, _ = connected_components(A, directed=True, connection="weak")
        n_strong, lab_s = connected_components(A, directed=True, connection="strong")
        largest_scc = np.bincount(lab_s).max() / n
        deg = np.bincount(arcs.ravel(), minlength=n)
        route_cont = float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0])) if len(route) > 1 else np.nan
        la, ld, _ = align_points_to_route(trip["xy"], trip["net"], route)
        labeldist_all[split].append(ld)
        rows.append({
            "split": split, "trip": tid,
            **{f"size_kb_{k}": round(v.stat().st_size / 1024, 1) if v.exists() else None for k, v in files.items()},
            "n_gps": len(raw), "n_nan_rows": n_nan, "n_non_increasing_t": int((dt_raw <= 0).sum()),
            "duration_s": float(tr[-1, 2] - tr[0, 2]), "pct_dt_1s": float(np.mean(dt == 1) * 100),
            "max_gap_s": float(dt.max()), "n_gaps_gt_10s": int((dt > 10).sum()),
            "length_km": float(step.sum() / 1000), "max_speed_kmh": float(speed.max() * 3.6),
            "n_jumps_gt_200kmh": int((speed * 3.6 > 200).sum()),
            "lon_min": tr[:, 0].min(), "lon_max": tr[:, 0].max(),
            "lat_min": tr[:, 1].min(), "lat_max": tr[:, 1].max(),
            "n_nodes": n, "n_arcs": len(arcs), "isolated_nodes": int((deg == 0).sum()),
            "weak_components": n_weak, "strong_components": n_strong,
            "largest_scc_frac": round(largest_scc, 3),
            "pct_two_way": round(float(trip["net"].has_twin.mean() * 100), 1),
            "median_arc_m": float(np.median(trip["net"].length)),
            "route_len_arcs": len(route), "route_continuity": route_cont,
            "route_km": float(trip["net"].length[route].sum() / 1000),
            "label_dist_p50": float(np.median(ld)), "label_dist_p95": float(np.percentile(ld, 95)),
            "pct_pts_gt50m_from_route": float(np.mean(ld > 50) * 100),
        })
        print(split, tid, len(raw), "pts")

inv = pd.DataFrame(rows)
inv.to_csv(config.RESULTS_DIR / "eda_inventory.csv", index=False)
summary = inv.groupby("split").agg(
    trips=("trip", "count"), gps_total=("n_gps", "sum"), gps_median=("n_gps", "median"),
    gps_min=("n_gps", "min"), gps_max=("n_gps", "max"), nan_rows=("n_nan_rows", "sum"),
    nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
    isolated_nodes=("isolated_nodes", "sum"), pct_dt_1s=("pct_dt_1s", "mean"),
    max_gap_s=("max_gap_s", "max"), route_continuity_min=("route_continuity", "min"),
    pct_pts_gt50m=("pct_pts_gt50m_from_route", "mean"))
summary.to_csv(config.RESULTS_DIR / "eda_summary.csv")
print(summary.T)

# ---------- plots ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for s, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
    sub = inv[inv.split == s]
    ax[0].hist(sub.n_gps, bins=20, alpha=.6, label=s, color=c)
    g = np.concatenate(gap_all[s]); ax[1].hist(np.clip(g, 0, 30), bins=30, alpha=.6, label=s, color=c, log=True)
    l = np.concatenate(labeldist_all[s]); ax[2].hist(np.clip(l, 0, 100), bins=50, alpha=.6, label=s, color=c, log=True)
ax[0].set(title="GPS points per trip", xlabel="points"); ax[1].set(title="Sampling interval (clipped 30 s)", xlabel="seconds")
ax[2].set(title="GPS-to-reference-arc distance (clipped 100 m)", xlabel="metres")
for a in ax: a.legend()
plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "eda_distributions.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(inv.n_nodes, inv.n_arcs, c=(inv.split == "train").map({True: "tab:blue", False: "tab:orange"}))
ax[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road network size per trip (blue=train, orange=student test)")
for s, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
    for _, r in inv[inv.split == s].iterrows():
        tdir = (config.TRAIN_DIR if s == "train" else config.STUDENT_TEST_DIR) / r.trip
        tr = np.loadtxt(tdir / f"{r.trip}.track")
        ax[1].plot(tr[::10, 0], tr[::10, 1], lw=.6, color=c, alpha=.6)
ax[1].set(xlabel="longitude", ylabel="latitude", title="All trajectories (every 10th point)")
plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "eda_network_and_trajectories.png", dpi=120); plt.close()

# Single-trip plot: GPS vs network vs reference route
tid = inv[inv.split == "train"].sort_values("n_gps").trip.iloc[len(inv[inv.split == "train"]) // 2]
trip = load_trip(config.TRAIN_DIR, tid)
nd, ar, rt, tr = trip["nodes"], trip["arcs"], trip["route"], trip["track"]
pad = 0.003
m = (nd[:, 0] > tr[:, 0].min() - pad) & (nd[:, 0] < tr[:, 0].max() + pad) & (nd[:, 1] > tr[:, 1].min() - pad) & (nd[:, 1] < tr[:, 1].max() + pad)
sel = m[ar[:, 0]] & m[ar[:, 1]]
plt.figure(figsize=(9, 9))
for a, b in ar[sel]:
    plt.plot(nd[[a, b], 0], nd[[a, b], 1], color="lightgrey", lw=.6)
for a, b in ar[rt]:
    plt.plot(nd[[a, b], 0], nd[[a, b], 1], color="green", lw=2.5)
plt.scatter(tr[:, 0], tr[:, 1], s=2, color="blue")
plt.title(f"Train trip {tid}: road network (grey), reference route (green), GPS (blue)")
plt.savefig(config.PLOTS_DIR / f"eda_trip_{tid}.png", dpi=120); plt.close()

# ---------- data quality log ----------
with open(config.RESULTS_DIR / "eda_quality_log.md", "w") as f:
    f.write("# Data quality log (generated by scripts/01_explore_data.py)\n\n")
    f.write(summary.T.to_markdown() if hasattr(summary, "to_markdown") else summary.T.to_string())
    f.write("\n\n## Trips with notable issues\n")
    for _, r in inv.iterrows():
        issues = []
        if r.n_nan_rows: issues.append(f"{r.n_nan_rows} NaN rows")
        if r.max_gap_s > 30: issues.append(f"max gap {r.max_gap_s:.0f}s")
        if r.n_jumps_gt_200kmh: issues.append(f"{r.n_jumps_gt_200kmh} jumps >200 km/h")
        if r.pct_pts_gt50m_from_route > 1: issues.append(f"{r.pct_pts_gt50m_from_route:.1f}% points >50 m from reference route")
        if r.route_continuity < 1: issues.append(f"reference route continuity {r.route_continuity:.3f}")
        if r.isolated_nodes: issues.append(f"{r.isolated_nodes} isolated nodes")
        if issues:
            f.write(f"- {r.split}/{r.trip}: " + "; ".join(issues) + "\n")

# ---------- Folium map ----------
try:
    from pathsnap.maps import trip_map
    for s, sdir in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        t0 = list_trips(sdir)[0]
        trip_map(load_trip(sdir, t0), title=f"{s} trip {t0}").save(str(config.MAPS_DIR / f"eda_{s}_{t0}.html"))
    print("Folium maps written to maps/")
except ImportError:
    print("folium not installed -> skipped maps (pip install folium)")
