"""
policy.py

Small, from-scratch NumPy policies + a REINFORCE-style (policy-gradient) update
rule. No PyTorch is used in this project: torch was unavailable in this
environment (see README "What this actually is" for why) and this machine had
under 1GB free disk, so instead of risking a failed/partial torch install we
implement the RLHF-style feedback loop by hand with linear/softmax policies and
manual score-function-estimator (REINFORCE) gradients. This is a real,
correctly-implemented policy-gradient method -- not a placeholder.

Three policy classes, one per action-space shape needed by the 7 tasks:

- GaussianPolicy   : continuous action a ~ N(w.x, sigma^2)         (regression)
- BernoulliPolicy  : binary action a ~ Bernoulli(sigmoid(w.x))     (binary cls)
- SoftmaxPolicy    : categorical action a ~ Categorical(softmax(W.x)) (multiclass)

Each policy supports:
    .act(X)              -> sampled actions + auxiliary info (probs) for reward fns
    .predict(X)           -> deterministic (mode) prediction, used for evaluation
    .reinforce_update(X, actions, rewards, lr) -> one policy-gradient step with a
                             running-mean baseline to reduce variance.
"""

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    e = np.exp(Z)
    return e / np.sum(e, axis=1, keepdims=True)


class GaussianPolicy:
    """Continuous-action policy for regression tasks: a ~ N(w.x, sigma^2)."""

    def __init__(self, dim, sigma=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.1, dim)
        self.b = 0.0
        self.sigma = sigma
        self.baseline = 0.0

    def mean(self, X):
        return X @ self.w + self.b

    def act(self, X, rng):
        mu = self.mean(X)
        a = mu + rng.normal(0, self.sigma, size=mu.shape)
        return a, {"mu": mu}

    def predict(self, X):
        return self.mean(X)

    def reinforce_update(self, X, actions, rewards, lr=0.02, baseline_decay=0.9):
        self.baseline = baseline_decay * self.baseline + (1 - baseline_decay) * np.mean(rewards)
        advantage = rewards - self.baseline
        mu = self.mean(X)
        # d/dw log N(a; mu, sigma^2) = (a - mu)/sigma^2 * x
        score = (actions - mu) / (self.sigma ** 2)
        grad_w = X.T @ (advantage * score) / len(X)
        grad_b = np.mean(advantage * score)
        self.w += lr * grad_w
        self.b += lr * grad_b


class BernoulliPolicy:
    """Binary-action policy: a ~ Bernoulli(sigmoid(w.x))."""

    def __init__(self, dim, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.1, dim)
        self.b = 0.0
        self.baseline = 0.0

    def prob(self, X):
        return _sigmoid(X @ self.w + self.b)

    def act(self, X, rng):
        p = self.prob(X)
        a = (rng.uniform(0, 1, size=p.shape) < p).astype(int)
        return a, {"prob": p}

    def predict(self, X):
        return (self.prob(X) >= 0.5).astype(int)

    def reinforce_update(self, X, actions, rewards, lr=0.05, baseline_decay=0.9):
        self.baseline = baseline_decay * self.baseline + (1 - baseline_decay) * np.mean(rewards)
        advantage = rewards - self.baseline
        p = self.prob(X)
        # d/dw log Bernoulli(a; p) = (a - p) * x   [with p = sigmoid(w.x)]
        score = actions - p
        grad_w = X.T @ (advantage * score) / len(X)
        grad_b = np.mean(advantage * score)
        self.w += lr * grad_w
        self.b += lr * grad_b


class SoftmaxPolicy:
    """Categorical-action policy: a ~ Categorical(softmax(W.x))."""

    def __init__(self, dim, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, (dim, n_classes))
        self.b = np.zeros(n_classes)
        self.n_classes = n_classes
        self.baseline = 0.0

    def probs(self, X):
        return _softmax(X @ self.W + self.b)

    def act(self, X, rng):
        P = self.probs(X)
        n = len(X)
        a = np.array([rng.choice(self.n_classes, p=P[i]) for i in range(n)])
        return a, {"probs": P}

    def predict(self, X):
        return np.argmax(self.probs(X), axis=1)

    def reinforce_update(self, X, actions, rewards, lr=0.05, baseline_decay=0.9):
        self.baseline = baseline_decay * self.baseline + (1 - baseline_decay) * np.mean(rewards)
        advantage = rewards - self.baseline
        P = self.probs(X)
        n = len(X)
        onehot = np.zeros((n, self.n_classes))
        onehot[np.arange(n), actions] = 1.0
        # d/dW log softmax(a) = (onehot(a) - P) outer x
        score = onehot - P  # (n, n_classes)
        grad_W = X.T @ (score * advantage[:, None]) / n
        grad_b = np.mean(score * advantage[:, None], axis=0)
        self.W += lr * grad_W
        self.b += lr * grad_b
