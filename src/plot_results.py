"""Create the score-comparison plots used in the paper.

The values are copied from the reproducible result tables in the paper and
the corresponding metric artifacts.  Keeping the plotting code in the repo
makes the figures easy to regenerate without calling any model or API.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"


def task2_model_comparison() -> None:
    models = ["Gemini 2.5 Pro RAG", "GLM-5.2 RAG", "GPT-5.4-mini VLM", "MiniLM + LR"]
    micro = np.array([0.6507, 0.6280, 0.6217, 0.4603])
    macro = np.array([0.6140, 0.5963, 0.5748, 0.4489])

    x = np.arange(len(models))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.15, 3.4))
    bars_micro = ax.bar(x - width / 2, micro, width, label="Micro F1", color="#4472C4")
    bars_macro = ax.bar(x + width / 2, macro, width, label="Macro F1", color="#ED7D31")
    ax.set_xticks(x, models, rotation=12, ha="right")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("F1 score")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    for bars in (bars_micro, bars_macro):
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    # Keep the metric legend below the plotting area so it never competes with
    # the bars or the complete figure caption in the paper.
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.015), frameon=False, ncol=2)
    fig.subplots_adjust(bottom=0.31)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(OUT / "task2_model_score_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def task1_retrieval_comparison() -> None:
    configs = [
        "MiniLM fused baseline",
        "BGE-M3 dense",
        "BGE-M3 dense+sparse",
        "BGE-M3 dense+sparse+ColBERT",
        "BGE-Reranker-v2-M3",
    ]
    # Scores on the 369 reconstructed groups with non-empty gold pages.
    scores = {
        "Hit@1": [0.220, 0.238, 0.249, 0.266, 0.160],
        "Hit@5": [0.493, 0.504, 0.523, 0.545, 0.504],
        "Hit@10": [0.631, 0.631, 0.648, 0.675, 0.672],
        "MRR": [0.350, 0.369, 0.381, 0.398, 0.313],
    }
    colors = ["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5"]

    x = np.arange(len(configs))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.15, 4.0))
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    for (metric, values), offset, color in zip(scores.items(), offsets, colors):
        bars = ax.bar(x + offset, values, width, label=metric, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=6.8)
    short_configs = [
        "MiniLM\nfused",
        "BGE-M3\ndense",
        "BGE-M3\ndense+sparse",
        "BGE-M3\ndense+sparse+\nColBERT",
        "BGE-\nReranker-v2-M3",
    ]
    ax.set_xticks(x, short_configs)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, ncol=4)
    fig.subplots_adjust(bottom=0.36)
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.savefig(OUT / "task1_retrieval_score_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    task2_model_comparison()
    task1_retrieval_comparison()
    print("Wrote Task 1 and Task 2 comparison plots to", OUT)
