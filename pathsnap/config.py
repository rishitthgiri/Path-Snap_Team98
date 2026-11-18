"""Central configuration. Paths can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
# Evaluator trips are only placed here on the API laptop on evaluation day.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DATA", DATA_DIR / "evaluator_test"))

RESULTS_DIR = ROOT / "results"
ARTIFACT_DIR = ROOT / "artifacts"
PLOTS_DIR = ROOT / "plots"
MAPS_DIR = ROOT / "maps"
CACHE_DIR = ROOT / "cache"

SEED = 42

# Candidate generation
CAND_RADIUS_M = 50.0   # search radius around each GPS point
CAND_MAX_K = 12        # keep at most K nearest arcs (twin arcs count separately)
DENSIFY_STEP_M = 10.0  # arc sampling step for the spatial index

# Feature engineering
HEADING_HALF_WINDOW = 3   # points either side used to estimate heading
CONTEXT_HALF_WINDOW = 5   # points either side used for context distance

# Training subsampling (1 Hz points are highly correlated)
TRAIN_POINT_STRIDE = 5

# HMM defaults (tuned by scripts/05_hmm_eval.py on train + student test only)
HMM_DEFAULTS = {
    "beta": 10.0,           # metres; transition scale |route dist - straight dist|
    "emission_weight": 1.0, # weight on log classifier probability
    "back_tol": 15.0,       # metres of backward jitter tolerated on the same arc
    "unreachable_logp": -40.0,
}
