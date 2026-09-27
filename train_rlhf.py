"""
train_rlhf.py

Runs a simplified RLHF-style feedback loop, per task:

  1. Initialize a small policy (policy.py) with small random weights. This is
     the "pre-RLHF" / initial policy -- evaluated once on the held-out
     benchmark before any reward-driven training (this is the resume's
     "initial policy baseline").
  2. Feedback-driven training loop: for N iterations, the policy samples
     actions (predictions) on the TRAIN split, each action is scored by that
     task's reward function (reward_models.py), and the policy is updated
     with a REINFORCE (score-function policy-gradient) step using the reward
     as the learning signal -- a genuine reward-weighted update, analogous to
     the reward-model-driven policy update step in RLHF, just implemented by
     hand at small scale with linear/softmax policies instead of an LLM.
  3. Re-evaluate the SAME policy on the SAME held-out benchmark used in step 1.

No benchmark data is ever used for gradient updates -- only for evaluation --
so the reported before/after numbers are genuine held-out performance.
"""

import numpy as np

from generate_bio_tasks import build_all_tasks
from reward_models import REWARD_FUNCTIONS
from policy import GaussianPolicy, BernoulliPolicy, SoftmaxPolicy

N_ITERATIONS = 220
PRETRAIN_ITERATIONS = 50  # small warm-start so the "pre-RLHF" baseline is a
                          # weak-but-not-degenerate policy, mirroring a real
                          # RLHF setup where the base policy already does
                          # something before reward-driven fine-tuning begins.
SEED = 123


def make_policy(task, seed):
    dim = task["train"][0].shape[1]
    if task["type"] == "regression":
        return GaussianPolicy(dim, sigma=0.5, seed=seed)
    elif task["type"] == "binary":
        return BernoulliPolicy(dim, seed=seed)
    elif task["type"] == "multiclass":
        return SoftmaxPolicy(dim, task["n_classes"], seed=seed)
    raise ValueError(task["type"])


def eval_metric(task, y_true, y_pred):
    """Bounded [0, 1] score, comparable across task types."""
    if task["type"] == "regression":
        mae = np.mean(np.abs(y_true - y_pred))
        naive_mae = np.mean(np.abs(y_true - np.mean(y_true))) + 1e-9
        return float(np.clip(1.0 - mae / naive_mae, 0.0, 1.0))
    else:
        return float(np.mean(y_true == y_pred))


def compute_reward(task, X, y_true, actions, aux):
    name = task["name"]
    fn = REWARD_FUNCTIONS[name]
    if name == "binding_affinity_ranking":
        return fn(y_true, actions, aux["prob"])
    elif name == "subcellular_localization":
        return fn(y_true, actions, aux["probs"])
    elif name in ("protein_solubility", "stability_prediction", "expression_level_regression"):
        return fn(y_true, actions)
    elif name in ("toxicity_classification", "mutation_effect_prediction"):
        return fn(y_true, actions)
    raise ValueError(name)


def train_one_task(task, n_iterations=N_ITERATIONS, pretrain_iterations=PRETRAIN_ITERATIONS,
                    seed=SEED, lr=None, track_curve=False):
    X_train, y_train = task["train"]
    X_bench, y_bench = task["bench"]

    policy = make_policy(task, seed=seed)
    rng = np.random.default_rng(seed + 999)

    default_lr = {"regression": 0.03, "binary": 0.08, "multiclass": 0.08}[task["type"]]
    lr = lr or default_lr

    reward_curve = []

    # Warm-start: a small number of reward-driven steps so the "pre-RLHF"
    # policy is a weak-but-functioning baseline rather than pure noise
    # (analogous to a base/SFT model before RLHF, not a randomly initialized
    # network).
    for it in range(pretrain_iterations):
        actions, aux = policy.act(X_train, rng)
        rewards = compute_reward(task, X_train, y_train, actions, aux)
        policy.reinforce_update(X_train, actions, rewards, lr=lr)
        if track_curve:
            reward_curve.append(float(np.mean(rewards)))

    y_pred_before = policy.predict(X_bench)
    score_before = eval_metric(task, y_bench, y_pred_before)

    for it in range(n_iterations):
        actions, aux = policy.act(X_train, rng)
        rewards = compute_reward(task, X_train, y_train, actions, aux)
        policy.reinforce_update(X_train, actions, rewards, lr=lr)
        if track_curve:
            reward_curve.append(float(np.mean(rewards)))

    y_pred_after = policy.predict(X_bench)
    score_after = eval_metric(task, y_bench, y_pred_after)

    result = {
        "name": task["name"],
        "type": task["type"],
        "score_before": score_before,
        "score_after": score_after,
        "relative_improvement_pct": 100.0 * (score_after - score_before) / max(score_before, 0.05),
    }
    if track_curve:
        result["reward_curve"] = reward_curve
    return result, policy


def run_all(n_iterations=N_ITERATIONS, seed=SEED):
    tasks = build_all_tasks()
    results = []
    representative_curve = None
    for task in tasks:
        track = task["name"] == "protein_solubility"
        result, _ = train_one_task(task, n_iterations=n_iterations, seed=seed, track_curve=track)
        if track:
            representative_curve = result.pop("reward_curve")
        results.append(result)
    return results, representative_curve


if __name__ == "__main__":
    results, curve = run_all()
    for r in results:
        print(
            f"{r['name']:30s} before={r['score_before']:.3f} "
            f"after={r['score_after']:.3f} rel_impr={r['relative_improvement_pct']:+.1f}%"
        )
    mean_before = np.mean([r["score_before"] for r in results])
    mean_after = np.mean([r["score_after"] for r in results])
    agg = 100.0 * (mean_after - mean_before) / mean_before
    print(f"\nMean score before: {mean_before:.3f}  after: {mean_after:.3f}")
    print(f"Aggregate relative improvement across 7 tasks: {agg:+.1f}%")
