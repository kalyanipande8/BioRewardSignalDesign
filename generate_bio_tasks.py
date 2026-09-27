"""
generate_bio_tasks.py

Generates 7 distinct SYNTHETIC biological prediction tasks. These are NOT real
biological datasets -- they are engineered feature vectors (standing in for the
kind of sequence/structure-derived descriptors a real bio-ML pipeline would use:
hydrophobicity indices, charge, length, motif counts, etc.) paired with targets
generated from a fixed, known data-generating process plus noise. This keeps the
project runnable on a laptop CPU in seconds while still being a REAL learning
problem (the generating function is not trivially linear-recoverable from a
glance, and there is genuine label noise).

Each task returns:
    train: (X_train, y_train)
    benchmark (held-out): (X_bench, y_bench)   <- never used for fitting/training

All randomness is seeded (SEED) for reproducibility.
"""

import numpy as np

SEED = 42
FEATURE_DIM = 12
N_TRAIN = 400
N_BENCH = 150


def _rng(offset):
    # distinct, reproducible sub-streams per task
    return np.random.default_rng(SEED + offset)


def _features(n, rng):
    """Synthetic 'sequence-derived descriptor' feature vectors."""
    return rng.normal(0, 1, size=(n, FEATURE_DIM))


# ---------------------------------------------------------------------------
# Task 1: Protein solubility prediction (regression, target in [0, 1])
# ---------------------------------------------------------------------------
def task_solubility(offset=1):
    rng = _rng(offset)
    w = rng.normal(0, 1, FEATURE_DIM)

    def gen(n):
        X = _features(n, rng)
        raw = X @ w + 0.4 * X[:, 0] * X[:, 1]  # nonlinear interaction term
        noise = rng.normal(0, 0.3, n)
        y = 1 / (1 + np.exp(-(raw + noise)))  # squash to [0, 1] solubility score
        return X, y

    X_train, y_train = gen(N_TRAIN)
    X_bench, y_bench = gen(N_BENCH)
    return {
        "name": "protein_solubility",
        "type": "regression",
        "train": (X_train, y_train),
        "bench": (X_bench, y_bench),
    }


# ---------------------------------------------------------------------------
# Task 2: Binding-affinity ranking (pairwise binary: does A bind tighter than B?)
# ---------------------------------------------------------------------------
def task_binding_ranking(offset=2):
    rng = _rng(offset)
    w = rng.normal(0, 1, FEATURE_DIM)

    def gen(n):
        Xa = _features(n, rng)
        Xb = _features(n, rng)
        affinity_a = Xa @ w + rng.normal(0, 0.5, n)
        affinity_b = Xb @ w + rng.normal(0, 0.5, n)
        y = (affinity_a > affinity_b).astype(int)  # 1 if A binds tighter
        X = Xa - Xb  # pairwise difference features
        return X, y

    X_train, y_train = gen(N_TRAIN)
    X_bench, y_bench = gen(N_BENCH)
    return {
        "name": "binding_affinity_ranking",
        "type": "binary",
        "train": (X_train, y_train),
        "bench": (X_bench, y_bench),
    }


# ---------------------------------------------------------------------------
# Task 3: Toxicity classification (binary, imbalanced ~20% toxic)
# ---------------------------------------------------------------------------
def task_toxicity(offset=3):
    rng = _rng(offset)
    w = rng.normal(0, 1, FEATURE_DIM)

    def gen(n):
        X = _features(n, rng)
        logits = X @ w - 1.0  # shift to make toxic (1) the minority class
        p = 1 / (1 + np.exp(-logits))
        y = (rng.uniform(0, 1, n) < p).astype(int)
        return X, y

    X_train, y_train = gen(N_TRAIN)
    X_bench, y_bench = gen(N_BENCH)
    return {
        "name": "toxicity_classification",
        "type": "binary",
        "train": (X_train, y_train),
        "bench": (X_bench, y_bench),
    }


# ---------------------------------------------------------------------------
# Task 4: Stability prediction (regression, ddG-like continuous value)
# ---------------------------------------------------------------------------
def task_stability(offset=4):
    rng = _rng(offset)
    w = rng.normal(0, 1, FEATURE_DIM)

    def gen(n):
        X = _features(n, rng)
        y = X @ w + 0.3 * np.sin(X[:, 2] * 2.0) + rng.normal(0, 0.4, n)
        return X, y

    X_train, y_train = gen(N_TRAIN)
    X_bench, y_bench = gen(N_BENCH)
    return {
        "name": "stability_prediction",
        "type": "regression",
        "train": (X_train, y_train),
        "bench": (X_bench, y_bench),
    }


# ---------------------------------------------------------------------------
# Task 5: Expression-level regression (log expression value)
# ---------------------------------------------------------------------------
def task_expression(offset=5):
    rng = _rng(offset)
    w = rng.normal(0, 1, FEATURE_DIM)

    def gen(n):
        X = _features(n, rng)
        y = 2.0 + X @ (0.5 * w) + rng.normal(0, 0.6, n)
        return X, y

    X_train, y_train = gen(N_TRAIN)
    X_bench, y_bench = gen(N_BENCH)
    return {
        "name": "expression_level_regression",
        "type": "regression",
        "train": (X_train, y_train),
        "bench": (X_bench, y_bench),
    }


# ---------------------------------------------------------------------------
# Task 6: Subcellular localization classification (4 classes)
# ---------------------------------------------------------------------------
def task_localization(offset=6, n_classes=4):
    rng = _rng(offset)
    W = rng.normal(0, 1, (FEATURE_DIM, n_classes))

    def gen(n):
        X = _features(n, rng)
        logits = X @ W + rng.normal(0, 0.5, (n, n_classes))
        y = np.argmax(logits, axis=1)
        return X, y

    X_train, y_train = gen(N_TRAIN)
    X_bench, y_bench = gen(N_BENCH)
    return {
        "name": "subcellular_localization",
        "type": "multiclass",
        "n_classes": n_classes,
        "train": (X_train, y_train),
        "bench": (X_bench, y_bench),
    }


# ---------------------------------------------------------------------------
# Task 7: Mutation-effect prediction (3 classes, imbalanced: mostly "neutral")
# ---------------------------------------------------------------------------
def task_mutation_effect(offset=7, n_classes=3):
    # classes: 0 = deleterious, 1 = neutral (majority), 2 = beneficial
    rng = _rng(offset)
    W = rng.normal(0, 1, (FEATURE_DIM, n_classes))
    bias = np.array([-0.3, 1.2, -0.3])  # bias towards "neutral"

    def gen(n):
        X = _features(n, rng)
        logits = X @ W + bias + rng.normal(0, 0.5, (n, n_classes))
        y = np.argmax(logits, axis=1)
        return X, y

    X_train, y_train = gen(N_TRAIN)
    X_bench, y_bench = gen(N_BENCH)
    return {
        "name": "mutation_effect_prediction",
        "type": "multiclass",
        "n_classes": n_classes,
        "train": (X_train, y_train),
        "bench": (X_bench, y_bench),
    }


TASK_BUILDERS = [
    task_solubility,
    task_binding_ranking,
    task_toxicity,
    task_stability,
    task_expression,
    task_localization,
    task_mutation_effect,
]


def build_all_tasks():
    """Returns a list of 7 task dicts, each with train + held-out benchmark data."""
    return [builder() for builder in TASK_BUILDERS]


if __name__ == "__main__":
    tasks = build_all_tasks()
    for t in tasks:
        Xtr, ytr = t["train"]
        Xb, yb = t["bench"]
        print(f"{t['name']:30s} type={t['type']:10s} train={Xtr.shape} bench={Xb.shape}")
