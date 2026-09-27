"""
reward_models.py

One reward function per biological prediction task. Each reward function scores
a policy's PREDICTION against the ground truth, and is deliberately designed
differently per task to reflect what actually matters for that kind of
biological readout -- this is the "reward signal design" piece of the project,
not a single copy-pasted MSE/accuracy formula repeated seven times.

Every reward function returns a value roughly in [-1, 1] (or [0, 1]) so
rewards are comparable in scale across tasks for the aggregate report, but the
SHAPE of each reward (what is rewarded/penalized, and how) differs:

1. protein_solubility        -> continuous closeness reward with a steep
                                 penalty that grows superlinearly far from the
                                 target (errors matter increasingly more).
2. binding_affinity_ranking  -> pairwise ranking correctness reward with a
                                 confidence/calibration bonus: rewards being
                                 right AND confident, penalizes being wrong
                                 AND confident (asymmetric, calibration-aware).
3. toxicity_classification   -> asymmetric cost-sensitive reward: missing a
                                 toxic compound (false negative) is penalized
                                 much more heavily than a false alarm, because
                                 in real safety screening false negatives are
                                 far more costly.
4. stability_prediction      -> thresholded "tolerance band" reward: full
                                 credit within a tolerance of the true ddG,
                                 partial credit that decays outside it (mirrors
                                 how stability calls are often bucketed).
5. expression_level_regression -> rank/correlation-flavored reward computed
                                 batch-wise (reward depends on whether the
                                 prediction preserves relative ordering vs.
                                 batch-mean, similar to how expression QC cares
                                 about relative levels more than absolute
                                 units).
6. subcellular_localization  -> correctness reward combined with an entropy
                                 (confidence-calibration) term: correct +
                                 confident is rewarded most, correct-but-unsure
                                 less, wrong-and-confident is punished hardest.
7. mutation_effect_prediction -> class-imbalance-aware reward: correct
                                 predictions on rare classes (deleterious /
                                 beneficial) are worth more than correct
                                 predictions on the majority "neutral" class,
                                 mirroring why rare-event detection matters
                                 more than baseline accuracy in mutation
                                 scanning.

Every reward_* function has the same call signature relevant to its task type
so train_rlhf.py can call them generically per task.
"""

import numpy as np


def reward_solubility(y_true, y_pred):
    """Continuous closeness reward with superlinear penalty for large error."""
    err = np.abs(y_true - y_pred)
    return 1.0 - np.clip(err ** 1.5 * 4.0, 0.0, 2.0)


def reward_binding_ranking(y_true, pred_label, pred_prob):
    """
    Correctness + calibration bonus/penalty.
    pred_prob: policy's probability that action==1 (A binds tighter).
    """
    correct = (pred_label == y_true).astype(float)
    confidence = np.abs(pred_prob - 0.5) * 2.0  # 0 (unsure) .. 1 (very sure)
    # correct & confident -> big reward; wrong & confident -> big penalty
    reward = np.where(
        correct == 1.0,
        0.5 + 0.5 * confidence,
        -0.5 - 0.5 * confidence,
    )
    return reward


def reward_toxicity(y_true, pred_label):
    """Cost-sensitive: false negatives (missed toxic=1) penalized hardest."""
    reward = np.zeros_like(y_true, dtype=float)
    tp = (y_true == 1) & (pred_label == 1)
    tn = (y_true == 0) & (pred_label == 0)
    fp = (y_true == 0) & (pred_label == 1)
    fn = (y_true == 1) & (pred_label == 0)
    reward[tp] = 1.0
    reward[tn] = 0.5
    reward[fp] = -0.3
    reward[fn] = -1.0  # missing real toxicity is the worst outcome
    return reward


def reward_stability(y_true, y_pred, tol=0.5):
    """Tolerance-band reward: full credit within tol, decaying outside it."""
    err = np.abs(y_true - y_pred)
    within = err <= tol
    reward = np.where(within, 1.0 - 0.4 * (err / tol), np.exp(-(err - tol)) - 0.4)
    return reward


def reward_expression(y_true, y_pred):
    """
    Mostly rank/relative-ordering reward: compares each prediction's deviation
    from the batch mean against the true deviation from the batch mean
    (captures whether relative expression level is preserved, similar to how
    expression QC cares about relative levels within a batch). A smaller
    absolute-error anchoring term is blended in so the signal isn't fully
    shift-invariant (a policy still has to get the absolute level roughly
    right, not just the relative ordering).
    """
    true_dev = y_true - np.mean(y_true)
    pred_dev = y_pred - np.mean(y_pred)
    same_sign = np.sign(true_dev) == np.sign(pred_dev)
    magnitude_err = np.abs(np.abs(true_dev) - np.abs(pred_dev))
    rank_term = np.where(same_sign, 0.6 - 0.3 * np.tanh(magnitude_err), -0.4)
    abs_term = 1.0 - np.tanh(np.abs(y_true - y_pred))
    reward = 0.6 * rank_term + 0.4 * abs_term
    return reward


def reward_localization(y_true, pred_label, pred_probs):
    """
    Correctness + entropy-based calibration term.
    pred_probs: (n, n_classes) softmax probabilities from the policy.
    """
    n = len(y_true)
    correct = (pred_label == y_true).astype(float)
    p_true_class = pred_probs[np.arange(n), y_true]
    eps = 1e-9
    entropy = -np.sum(pred_probs * np.log(pred_probs + eps), axis=1)
    max_entropy = np.log(pred_probs.shape[1])
    confidence = 1.0 - entropy / max_entropy  # 0 (uniform) .. 1 (certain)
    reward = np.where(
        correct == 1.0,
        0.4 + 0.6 * confidence,
        -0.2 - 0.6 * confidence,  # confidently wrong is punished hardest
    )
    # small extra shaping from probability mass assigned to the true class
    reward = reward + 0.1 * (p_true_class - 1.0 / pred_probs.shape[1])
    return reward


def reward_mutation_effect(y_true, pred_label, class_weights=None):
    """
    Class-imbalance-aware reward: correct calls on rare classes (0=deleterious,
    2=beneficial) score higher than correct calls on the majority class
    (1=neutral); incorrect calls on rare classes are also penalized more.
    """
    if class_weights is None:
        # neutral (1) is the common/easy class -> lower weight
        class_weights = {0: 1.5, 1: 0.6, 2: 1.5}
    correct = (pred_label == y_true)
    weights = np.array([class_weights[int(c)] for c in y_true])
    reward = np.where(correct, weights, -0.5 * weights)
    return reward


REWARD_FUNCTIONS = {
    "protein_solubility": reward_solubility,
    "binding_affinity_ranking": reward_binding_ranking,
    "toxicity_classification": reward_toxicity,
    "stability_prediction": reward_stability,
    "expression_level_regression": reward_expression,
    "subcellular_localization": reward_localization,
    "mutation_effect_prediction": reward_mutation_effect,
}
