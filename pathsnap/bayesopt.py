"""Minimal Bayesian optimisation: Gaussian-process surrogate + Expected Improvement.

Implemented with scikit-learn's GaussianProcessRegressor so it runs without extra
packages. Search dimensions are mapped to the unit cube; 'log' dims are searched
in log space, 'int' dims are rounded.
"""
import warnings
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class Space:
    def __init__(self, dims):
        self.dims = dims  # list of (name, low, high, kind) kind in {"float","log","int"}

    def decode(self, u):
        out = {}
        for (name, lo, hi, kind), x in zip(self.dims, u):
            if kind == "log":
                v = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))))
            else:
                v = lo + x * (hi - lo)
            out[name] = int(round(v)) if kind == "int" else float(v)
        return out


def bayes_opt(objective, space, n_init=6, n_iter=19, seed=42, n_cand=4000, log=print):
    """Maximise objective(params)->float. Returns list of trial dicts."""
    rng = np.random.default_rng(seed)
    d = len(space.dims)
    U, Y, trials = [], [], []
    for it in range(n_init + n_iter):
        if it < n_init:
            u = rng.random(d); how = "random"
        else:
            k = ConstantKernel(1.0) * Matern(length_scale=np.full(d, 0.3), nu=2.5) + WhiteKernel(1e-4)
            gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(np.array(U), np.array(Y))
            C = rng.random((n_cand, d))
            mu, sd = gp.predict(C, return_std=True)
            best = max(Y); xi = 0.01 * (np.std(Y) + 1e-9)
            z = (mu - best - xi) / np.maximum(sd, 1e-9)
            ei = (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)
            u = C[int(np.argmax(ei))]; how = "GP-EI"
        params = space.decode(u)
        val, info = objective(params)
        U.append(u); Y.append(val)
        trials.append({"trial": it, "strategy": how, **params, "objective": val, **info})
        log(f"trial {it:02d} [{how}] {params} -> {val:.5f} (best {max(Y):.5f})")
    return trials
