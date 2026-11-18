"""Evaluation metrics: observation-level Top-1, candidate miss rate, route continuity,
route-level precision/recall/F1."""
import numpy as np
import pandas as pd


def top1_choice(df, score_col="prob"):
    """Per (trip, pt) pick the candidate arc with the highest score."""
    idx = df.groupby(["trip", "pt"], sort=False)[score_col].idxmax()
    return df.loc[idx.values, ["trip", "pt", "arc"]].rename(columns={"arc": "pred_arc"})


def observation_top1(points, choice):
    """points: reliable point table with label_arc. Points without candidates count as wrong."""
    m = points.merge(choice, on=["trip", "pt"], how="left")
    return float((m.pred_arc == m.label_arc).mean())


def collapse(seq):
    seq = [int(a) for a in seq if a is not None and a >= 0]
    out = []
    for a in seq:
        if not out or out[-1] != a:
            out.append(a)
    return out


def route_continuity(arc_seq, arcs):
    """Fraction of consecutive distinct arcs in the sequence that are directly connected
    (to_node of a == from_node of b). 1.0 = a drivable, gap-free route."""
    s = collapse(arc_seq)
    if len(s) < 2:
        return 1.0
    s = np.asarray(s)
    return float(np.mean(arcs[s[:-1], 1] == arcs[s[1:], 0]))


def route_prf(pred_route, ref_route, arc_len):
    """Length-weighted precision/recall/F1 between predicted and reference arc sets."""
    P, R = set(map(int, pred_route)), set(map(int, ref_route))
    inter = P & R
    lp = sum(arc_len[a] for a in P) or 1e-9
    lr = sum(arc_len[a] for a in R) or 1e-9
    li = sum(arc_len[a] for a in inter)
    p, r = li / lp, li / lr
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"route_precision": float(p), "route_recall": float(r), "route_f1": float(f)}
