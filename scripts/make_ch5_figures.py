#!/usr/bin/env python3
"""Generate the figures for Chapter 5.

    python scripts/make_ch5_figures.py --data-dir data --output-dir figures

Figure 5.1 is drawn from the architecture itself and needs no inputs. Figure 5.2
reads the operating-point sweep produced by sweep_operating_points.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})


def figure_architecture(output_dir: Path) -> None:
    """The three agents and the artefacts between them."""
    fig, axis = plt.subplots(figsize=(6.6, 3.4))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 5.2)
    axis.axis("off")
    axis.grid(False)

    def box(x, y, width, height, title, subtitle, fill):
        axis.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.08",
            facecolor=fill, edgecolor="black", linewidth=1.0,
        ))
        axis.text(x + width / 2, y + height * 0.62, title, ha="center",
                  va="center", fontsize=9, fontweight="bold")
        axis.text(x + width / 2, y + height * 0.26, subtitle, ha="center",
                  va="center", fontsize=7.5, style="italic")

    def arrow(x1, y1, x2, y2, label):
        axis.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
            linewidth=1.0, color="black",
        ))
        if label:
            # Rotated and set well above the arrow, since the gap between boxes
            # is narrower than the filename it carries.
            axis.text((x1 + x2) / 2, y1 + 0.72, label, ha="center", va="bottom",
                      fontsize=6.5, family="monospace", rotation=90)

    # Inputs, produced by the stages of Chapters 3 and 4.
    box(0.1, 3.5, 2.5, 1.3, "Scored images",
        "IQA metrics, embeddings,\nprovenance", "#E8E8E8")

    box(0.1, 1.0, 2.6, 1.35, "Data Inspector",
        "thresholds, decisions,\nconfidence bands", "#D6E4F0")
    box(3.7, 1.0, 2.6, 1.35, "Dataset Summary",
        "aggregation, class balance,\nrecommendations", "#D6E4F0")
    box(7.3, 1.0, 2.6, 1.35, "LLM Explainer",
        "audit report,\nfactual consistency", "#D6E4F0")

    arrow(1.40, 3.5, 1.40, 2.40, "")
    arrow(2.75, 1.67, 3.65, 1.67, "inspection.json")
    arrow(6.35, 1.67, 7.25, 1.67, "summary.json")

    # The orchestrator spans all three rather than sitting between them.
    axis.add_patch(FancyBboxPatch(
        (0.1, 0.02), 9.8, 0.6, boxstyle="round,pad=0.06",
        facecolor="#F2F2F2", edgecolor="black", linewidth=0.8, linestyle="--",
    ))
    axis.text(5.0, 0.32,
              "Orchestrator: run manifest, provenance, partial-failure handling",
              ha="center", va="center", fontsize=8)

    path = output_dir / "fig5_1_architecture.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"    wrote {path}")


def figure_operating_points(data_dir: Path, output_dir: Path, dataset: str) -> None:
    """False rejection against recall, with severity-1 recall shown separately."""
    path = data_dir / "logs" / f"{dataset}_operating_points.csv"
    if not path.is_file():
        print(f"    {path.name} not found; run sweep_operating_points.py first")
        return
    frame = pd.read_csv(path).sort_values("false_reject_rate")

    fig, axis = plt.subplots(figsize=(6.2, 3.4))
    axis.plot(frame["false_reject_rate"], frame["recall_any_action"],
              marker="o", markersize=4, linewidth=1.2, color="#4C72B0",
              label="all severities")
    severity_column = next(
        (c for c in frame.columns if c.endswith("severity_1")), None
    )
    if severity_column:
        axis.plot(frame["false_reject_rate"], frame[severity_column],
                  marker="s", markersize=4, linewidth=1.2, linestyle="--",
                  color="#DD8452", label="severity 1 only")
    severe_column = next(
        (c for c in frame.columns if c.endswith("severity_3")), None
    )
    if severe_column:
        axis.plot(frame["false_reject_rate"], frame[severe_column],
                  marker="^", markersize=4, linewidth=1.0, linestyle=":",
                  color="#55A868", label="severity 3 only")

    axis.set_xscale("log")
    axis.set_xlabel("realised false-rejection rate (log scale)")
    axis.set_ylabel("recall of degraded images")
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title("Data discarded against defects detected")
    fig.tight_layout()
    out = output_dir / "fig5_2_operating_points.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"    wrote {out}")


def figure_redundancy(data_dir: Path, output_dir: Path, dataset: str) -> None:
    """Jaccard co-occurrence between checks, from the Summary Agent's output.

    The point of the figure is the contrast: two pairs at 1.00 against a
    background of low values. Showing only the saturated pairs would make the
    result look asserted rather than measured.
    """
    path = data_dir / "logs" / f"{dataset}_summary.json"
    if not path.is_file():
        print(f"    {path.name} not found; run run_summary.py first")
        return
    payload = json.loads(path.read_text())
    cooccurrence = payload.get("issue_cooccurrence", {})
    pairs = cooccurrence.get("top_cooccurring_pairs", [])
    fired = cooccurrence.get("checks_fired", {})
    if not pairs or not fired:
        print("    summary has no co-occurrence data")
        return

    # Order by how often each check fires, so the busiest sit together.
    checks = [name for name, _ in sorted(fired.items(), key=lambda kv: -kv[1])]
    matrix = np.full((len(checks), len(checks)), np.nan)
    position = {name: i for i, name in enumerate(checks)}
    for entry in pairs:
        a, b = entry.get("check_a"), entry.get("check_b")
        if a in position and b in position:
            matrix[position[a], position[b]] = entry["jaccard"]
            matrix[position[b], position[a]] = entry["jaccard"]
    for i in range(len(checks)):
        matrix[i, i] = 1.0

    fig, axis = plt.subplots(figsize=(5.4, 4.6))
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(checks)))
    axis.set_xticklabels([c.replace("_", " ") for c in checks], rotation=90)
    axis.set_yticks(range(len(checks)))
    axis.set_yticklabels([c.replace("_", " ") for c in checks])
    axis.grid(False)
    for i in range(len(checks)):
        for j in range(len(checks)):
            if i == j or not np.isfinite(matrix[i, j]):
                continue
            value = matrix[i, j]
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if value > 0.55 else "black")
    fig.colorbar(image, ax=axis, shrink=0.75, label="Jaccard coefficient")
    axis.set_title("Co-occurrence between checks")
    fig.tight_layout()
    out = output_dir / "fig5_3_redundancy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"    wrote {out}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dataset", default="chestxray14")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Figure 5.1 - agent architecture")
    figure_architecture(args.output_dir)
    print("Figure 5.2 - operating-point trade-off")
    figure_operating_points(args.data_dir, args.output_dir, args.dataset)
    print("Figure 5.3 - check redundancy")
    figure_redundancy(args.data_dir, args.output_dir, args.dataset)


if __name__ == "__main__":
    main()