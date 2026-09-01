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

    y = np.arange(len(models))
    height = 0.34
    fig, ax = plt.subplots(figsize=(7.15, 3.0))
    bars_micro = ax.barh(y + height / 2, micro, height, label="Micro F1", color="#4472C4")
    bars_macro = ax.barh(y - height / 2, macro, height, label="Macro F1", color="#ED7D31")
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("F1 score")
    ax.set_title("Task 2 model comparison (793 non-Chinese test rows)", pad=8)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
        ncol=2,
    )
    for bars in (bars_micro, bars_macro):
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    fig.tight_layout()
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

    y = np.arange(len(configs))
    height = 0.18
    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * height
    for (metric, values), offset, color in zip(scores.items(), offsets, colors):
        bars = ax.barh(y + offset, values, height, label=metric, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=6.8)
    ax.set_yticks(y, configs)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Score")
    ax.set_title("Task 1 retrieval comparison (369 non-empty-gold groups)", pad=8)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
        ncol=4,
    )
    fig.tight_layout()
    fig.savefig(OUT / "task1_retrieval_score_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    task2_model_comparison()
    task1_retrieval_comparison()
    print("Wrote Task 1 and Task 2 comparison plots to", OUT)
