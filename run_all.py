"""
run_all.py

Orchestrates the full pipeline:
  generate 7 tasks -> baseline ("pre-RLHF") eval -> reward-driven training
  -> final eval -> aggregate report -> figures.

Writes:
  output/report.md
  output/figures/before_after_bar.png
  output/figures/reward_curve.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_rlhf import run_all, N_ITERATIONS, PRETRAIN_ITERATIONS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def make_bar_chart(results, path):
    names = [r["name"] for r in results]
    before = [r["score_before"] for r in results]
    after = [r["score_after"] for r in results]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, before, width, label="Pre-RLHF (baseline)", color="#94a3b8")
    ax.bar(x + width / 2, after, width, label="Post-RLHF (reward-trained)", color="#2563eb")
    ax.set_ylabel("Held-out benchmark score (0-1)")
    ax.set_title("Per-task benchmark performance before vs. after reward-driven training")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_reward_curve(curve, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve, color="#059669", linewidth=1.5)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean batch reward")
    ax.set_title("Reward over training iterations (protein_solubility, representative task)")
    ax.axvline(PRETRAIN_ITERATIONS, color="#94a3b8", linestyle="--",
               label="end of pre-RLHF warm-start")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(results, agg, path):
    lines = []
    lines.append("# BioRewardSignalDesign -- Run Report\n")
    lines.append(
        "Real numbers from an actual run of `run_all.py`. See README.md for the full "
        "project description, reward-signal design rationale, and honest scope notes.\n"
    )
    lines.append("## Per-task results (held-out benchmark)\n")
    lines.append("| Task | Type | Before | After | Relative improvement |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['type']} | {r['score_before']:.3f} | "
            f"{r['score_after']:.3f} | {r['relative_improvement_pct']:+.1f}% |"
        )
    lines.append("")
    mean_before = np.mean([r["score_before"] for r in results])
    mean_after = np.mean([r["score_after"] for r in results])
    lines.append(f"**Mean score before:** {mean_before:.3f}  ")
    lines.append(f"**Mean score after:** {mean_after:.3f}  ")
    lines.append(f"**Aggregate relative improvement across 7 tasks: {agg:+.1f}%**\n")
    lines.append("## Figures\n")
    lines.append("![Before/after per task](figures/before_after_bar.png)\n")
    lines.append("![Reward curve](figures/reward_curve.png)\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    results, curve = run_all(n_iterations=N_ITERATIONS)

    print("=== Per-task results (held-out benchmark) ===")
    for r in results:
        print(
            f"{r['name']:30s} type={r['type']:10s} "
            f"before={r['score_before']:.3f} after={r['score_after']:.3f} "
            f"rel_impr={r['relative_improvement_pct']:+.1f}%"
        )

    mean_before = np.mean([r["score_before"] for r in results])
    mean_after = np.mean([r["score_after"] for r in results])
    agg = 100.0 * (mean_after - mean_before) / mean_before
    print(f"\nMean score before: {mean_before:.3f}  after: {mean_after:.3f}")
    print(f"Aggregate relative improvement across 7 tasks: {agg:+.1f}%")

    make_bar_chart(results, os.path.join(FIG_DIR, "before_after_bar.png"))
    make_reward_curve(curve, os.path.join(FIG_DIR, "reward_curve.png"))
    write_report(results, agg, os.path.join(OUT_DIR, "report.md"))
    print(f"\nWrote {os.path.join(OUT_DIR, 'report.md')} and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
