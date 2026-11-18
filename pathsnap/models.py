"""Candidate classifiers compared in the study (scikit-learn only)."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from .config import SEED

MODEL_SPECS = {
    "LogisticRegression": ("Linear / probabilistic", "C=1.0, L2, lbfgs, StandardScaler",
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
    "GaussianNB": ("Linear / probabilistic", "var_smoothing=1e-9",
        lambda: make_pipeline(StandardScaler(), GaussianNB())),
    "KNN": ("Instance-based", "k=25, distance-weighted, StandardScaler",
        lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=2))),
    "DecisionTree": ("Tree-based", "max_depth=12, min_samples_leaf=20",
        lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED)),
    "RandomForest": ("Ensemble (bagging)", "200 trees, min_samples_leaf=5",
        lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=2, random_state=SEED)),
    "ExtraTrees": ("Ensemble (bagging)", "200 trees, min_samples_leaf=5",
        lambda: ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=2, random_state=SEED)),
    "HistGradientBoosting": ("Boosting", "lr=0.1, max_iter=300, max_leaf_nodes=31, early stopping",
        lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                               early_stopping=True, random_state=SEED)),
    "MLP": ("Neural network", "64-32 ReLU, adam, alpha=1e-4, early stopping",
        lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                              batch_size=512, max_iter=60, early_stopping=True, random_state=SEED))),
}
