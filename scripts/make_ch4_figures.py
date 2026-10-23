#!/usr/bin/env python3
"""Generate the figures for Chapter 4 from the project's own log files.

    python scripts/make_ch4_figures.py --data-dir data --output-dir figures

Each figure is produced independently; a missing input causes that figure to be
skipped with an explanation rather than aborting the run. Where a log file has
an unexpected shape the script says so and names the file, so the mismatch can
be corrected rather than silently producing an empty plot.

Figures produced:

    fig4_1_metric_defect_heatmap   AUROC of every metric against every defect
    fig4_2_paired_comparison       paired bootstrap differences, forest plot
    fig4_3_fusion_strategies       fusion strategies against the best single scorer
    fig4_4_feature_ablation        performance by feature set
    fig4_5_coefficients            fitted logistic coefficients
    fig4_6_transfer                clean-score distributions and threshold transfer
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

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

LABELS = {"chestxray14": "ChestX-ray14", "isic": "ISIC"}


def tidy(name: str) -> str:
    return str(name).replace("_", " ")


def note(message: str) -> None:
    print(f"    {message}")


# ---------------------------------------------------------------- figure 4.1
def figure_heatmap(data_dir: Path, output_dir: Path, datasets: list[str]) -> None:
    """AUROC of every metric against every distortion, both datasets."""
    frames: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        for name in (
            f"{dataset}_iqa_auroc_valtest.csv", f"{dataset}_iqa_auroc.csv",
        ):
            path = data_dir / "logs" / name
            if path.is_file():
                frames[dataset] = pd.read_csv(path, index_col=0)
                break
        else:
            note(f"skip {dataset}: no IQA AUROC matrix found in {data_dir/'logs'}")

    if not frames:
        note("figure 4.1 skipped")
        return

    # Drop metrics with no directional score, which would otherwise occupy an
    # empty band across every panel.
    frames = {
        name: matrix.dropna(how="all")
        for name, matrix in frames.items()
    }

    fig, axes = plt.subplots(
        1, len(frames), figsize=(4.6 * len(frames), 4.0), squeeze=False,
        sharey=True,
    )
    for index, (axis, (dataset, matrix)) in enumerate(zip(axes[0], frames.items())):
        # A diverging map centred at 0.5 so that below-chance cells are visually
        # distinct from merely weak ones - the confounds are the point of this
        # figure, not the successes.
        image = axis.imshow(
            matrix.values.astype(float), cmap="RdBu_r", vmin=0.0, vmax=1.0,
            aspect="auto",
        )
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels([tidy(c) for c in matrix.columns], rotation=90)
        if index == 0:
            axis.set_yticks(range(len(matrix.index)))
            axis.set_yticklabels([tidy(i) for i in matrix.index])
        axis.set_title(LABELS.get(dataset, dataset))
        axis.grid(False)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.values[row, column]
                if not np.isfinite(value):
                    continue
                axis.text(
                    column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if abs(value - 0.5) > 0.32 else "black",
                )
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.7, label="AUROC")
    path = output_dir / "fig4_1_metric_defect_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"    wrote {path}")


# ---------------------------------------------------------------- figure 4.2
def figure_paired(data_dir: Path, output_dir: Path, dataset: str) -> None:
    """Forest plot of the paired bootstrap difference per distortion."""
    path = data_dir / "logs" / f"{dataset}_headtohead.csv"
    if not path.is_file():
        note(
            f"{path.name} not found. This figure needs a per-defect table with "
            "columns defect, diff, ci_low, ci_high. If the head-to-head result "
            "lives elsewhere, point this function at it."
        )
        return
    frame = pd.read_csv(path)
    required = {"diff", "ci_low", "ci_high"}
    if not required <= set(frame.columns):
        note(f"{path.name} lacks columns {sorted(required - set(frame.columns))}")
        return

    defect_column = "defect" if "defect" in frame.columns else frame.columns[0]
    frame = frame.sort_values("diff")
    positions = np.arange(len(frame))

    fig, axis = plt.subplots(figsize=(6.4, 3.6))
    significant = (frame["ci_low"] > 0) | (frame["ci_high"] < 0)
    axis.errorbar(
        frame["diff"], positions,
        xerr=[frame["diff"] - frame["ci_low"], frame["ci_high"] - frame["diff"]],
        fmt="o", markersize=4, capsize=3, linewidth=1.0, color="black",
    )
    # Hollow markers for the non-significant cases, so the two intervals that
    # span zero are distinguishable in greyscale.
    axis.scatter(
        frame.loc[~significant, "diff"], positions[~significant.to_numpy()],
        s=42, facecolors="white", edgecolors="black", zorder=3,
    )
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels([tidy(v) for v in frame[defect_column]])
    axis.set_xlabel(
        "difference in AUROC, embedding minus classical\n"
        "(negative favours classical metrics)"
    )
    axis.set_title("Paired bootstrap comparison per distortion")
    fig.tight_layout()
    out = output_dir / "fig4_2_paired_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"    wrote {out}")


# ---------------------------------------------------------------- figure 4.3
def _fusion_summary(data_dir: Path, dataset: str, backbone: str) -> dict | None:
    path = data_dir / "logs" / f"{dataset}_fusion_summary_{backbone}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def figure_fusion(
    data_dir: Path, output_dir: Path, dataset: str, backbone: str
) -> None:
    """Fusion strategies against the best single scorer on the aggregate task."""
    payload = _fusion_summary(data_dir, dataset, backbone)
    if payload is None:
        note(f"no fusion summary for {dataset}/{backbone}")
        return
    aggregate = payload.get("aggregate_clean_vs_any_degraded")
    if not isinstance(aggregate, dict):
        note("fusion summary has no aggregate_clean_vs_any_degraded block")
        return

    baseline = aggregate.get("best_single_auroc") or aggregate.get("baseline_auroc")
    # The summary stores one flat key per strategy and quantity, such as
    # "logistic_auroc" and "logistic_ci_low", rather than a nested block per
    # strategy, so the strategy names are recovered from the key suffixes.
    strategies = sorted(
        key[: -len("_auroc")] for key in aggregate
        if key.endswith("_auroc") and key != "best_single_auroc"
    )
    rows = []
    for strategy in strategies:
        auroc = aggregate.get(f"{strategy}_auroc")
        if auroc is None:
            continue
        rows.append({
            "strategy": strategy,
            "auroc": float(auroc),
            "diff": aggregate.get(f"{strategy}_diff"),
            "ci_low": aggregate.get(f"{strategy}_ci_low"),
            "ci_high": aggregate.get(f"{strategy}_ci_high"),
            "significant": aggregate.get(f"{strategy}_significant"),
        })
    if not rows:
        note("no per-strategy entries found in the aggregate block")
        return

    frame = pd.DataFrame(rows).sort_values("auroc")
    fig, axis = plt.subplots(figsize=(6.0, 3.0))
    positions = np.arange(len(frame))
    axis.barh(positions, frame["auroc"], color="#4C72B0", edgecolor="black",
              linewidth=0.4)
    if baseline is not None:
        axis.axvline(float(baseline), color="black", linestyle="--",
                     linewidth=1.0, label=f"best single scorer ({float(baseline):.3f})")
        axis.legend(loc="lower right", frameon=False)
    axis.set_yticks(positions)
    labels = []
    for _, row in frame.iterrows():
        diff = row.get("diff")
        if diff is None or pd.isna(diff):
            labels.append(tidy(row["strategy"]))
        else:
            marker = "*" if row.get("significant") else ""
            labels.append(f"{tidy(row['strategy'])}\n({diff:+.3f}{marker})")
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("AUROC, clean against any degraded")
    axis.set_xlim(min(0.5, frame["auroc"].min() - 0.05), 1.0)
    axis.set_title(f"{LABELS.get(dataset, dataset)}, {backbone}")
    for position, value in zip(positions, frame["auroc"]):
        axis.text(value + 0.005, position, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    out = output_dir / "fig4_3_fusion_strategies.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"    wrote {out}")


# ---------------------------------------------------------------- figure 4.4
def figure_ablation(data_dir: Path, output_dir: Path, datasets: list[str],
                    backbones: list[str]) -> None:
    """Aggregate performance by feature set, across datasets and backbones."""
    rows = []
    for dataset in datasets:
        for backbone in backbones:
            payload = _fusion_summary(data_dir, dataset, backbone)
            if payload is None:
                continue
            ablation = payload.get("feature_set_ablation", {})
            entry = ablation.get("logistic")
            if not isinstance(entry, dict):
                continue
            rows.append({
                "config": f"{LABELS.get(dataset, dataset)}\n{backbone}",
                "classical only": entry.get("auroc_iqa_only"),
                "embeddings only": entry.get("auroc_embeddings_only"),
                "all features": entry.get("auroc_all"),
            })
    if not rows:
        note("no feature_set_ablation blocks found; run evaluate_fusion.py first")
        return

    frame = pd.DataFrame(rows).set_index("config")
    fig, axis = plt.subplots(figsize=(6.6, 3.4))
    width = 0.26
    positions = np.arange(len(frame))
    styles = [("classical only", "#4C72B0", ""), ("embeddings only", "#DD8452", "///"),
              ("all features", "#55A868", "...")]
    for offset, (column, colour, hatch) in zip((-width, 0, width), styles):
        axis.bar(positions + offset, frame[column].astype(float), width,
                 label=column, color=colour, hatch=hatch, edgecolor="black",
                 linewidth=0.4)
    axis.set_xticks(positions)
    axis.set_xticklabels(frame.index, fontsize=8)
    axis.set_ylabel("AUROC, clean against any degraded")
    axis.set_ylim(0.5, 1.0)
    axis.legend(frameon=False, ncol=3, loc="upper center",
                bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    out = output_dir / "fig4_4_feature_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"    wrote {out}")


# ---------------------------------------------------------------- figure 4.5
def figure_coefficients(data_dir: Path, output_dir: Path, datasets: list[str],
                        backbone: str) -> None:
    """Fitted logistic coefficients, which carry the mechanism argument."""
    series: dict[str, pd.Series] = {}
    for dataset in datasets:
        payload = _fusion_summary(data_dir, dataset, backbone)
        if payload is None:
            continue
        by_set = payload.get("logistic_coefficients_by_feature_set", {})
        coefficients = by_set.get("all") if isinstance(by_set, dict) else None
        if coefficients is None:
            described = payload.get("strategies", {}).get("logistic", {})
            coefficients = described.get("coefficients")
        if isinstance(coefficients, dict) and coefficients:
            series[dataset] = pd.Series(coefficients, dtype=float)
    if not series:
        note("no logistic coefficients found in the fusion summaries")
        return

    frame = pd.DataFrame(series).dropna(how="all")
    order = frame.abs().max(axis=1).sort_values().index
    frame = frame.loc[order]

    fig, axis = plt.subplots(figsize=(6.4, max(3.0, 0.28 * len(frame))))
    positions = np.arange(len(frame))
    width = 0.38
    offsets = np.linspace(-width / 2, width / 2, max(len(frame.columns), 1))
    for offset, dataset in zip(offsets, frame.columns):
        axis.barh(positions + offset, frame[dataset], width / max(len(frame.columns), 1) * 1.6,
                  label=LABELS.get(dataset, dataset), edgecolor="black",
                  linewidth=0.4)
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.set_yticks(positions)
    axis.set_yticklabels([tidy(i) for i in frame.index], fontsize=8)
    axis.set_xlabel(
        "fitted coefficient on oriented scores\n"
        "(positive: a higher score is evidence of degradation)"
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    out = output_dir / "fig4_5_coefficients.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"    wrote {out}")


# ---------------------------------------------------------------- figure 4.6
def figure_transfer(data_dir: Path, output_dir: Path, datasets: list[str],
                    backbone: str) -> None:
    """Clean-score distributions, showing why a threshold cannot transfer."""
    path = data_dir / "logs" / f"transfer_{'_'.join(datasets)}_{backbone}.json"
    if not path.is_file():
        note(f"{path.name} not found; run evaluate_transfer.py first")
        return
    payload = json.loads(path.read_text())
    distributions = payload.get("results", {}).get("clean_score_distributions", {})
    thresholds = payload.get("results", {}).get("threshold_transfer", {})
    if not distributions:
        note("transfer log has no clean_score_distributions block")
        return

    fig, axis = plt.subplots(figsize=(6.4, 3.2))
    colours = {"chestxray14": "#4C72B0", "isic": "#DD8452"}
    for dataset, entry in distributions.items():
        mean = entry.get("clean_mean")
        sd = entry.get("clean_sd")
        if mean is None or sd is None:
            continue
        # The log records summary statistics rather than the raw scores, so the
        # clean population is drawn as a Gaussian with the recorded moments. This
        # is an approximation and is stated as such in the caption.
        grid = np.linspace(mean - 4 * sd, mean + 4 * sd, 400)
        density = np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        axis.plot(grid, density, color=colours.get(dataset, "black"),
                  label=f"{LABELS.get(dataset, dataset)} clean (mean {mean:.3f})")
        axis.fill_between(grid, density, alpha=0.18,
                          color=colours.get(dataset, "black"))

    for key, entry in thresholds.items():
        if key.startswith(f"from_{datasets[0]}"):
            value = entry.get("threshold")
            if value is not None:
                axis.axvline(float(value), color="black", linestyle="--",
                             linewidth=1.0,
                             label=f"threshold from {LABELS.get(datasets[0], datasets[0])}")
            break

    axis.set_xlabel("fused quality score (higher indicates worse quality)")
    axis.set_ylabel("density")
    axis.legend(frameon=False, fontsize=7)
    axis.set_title("Clean-image score distributions across modalities")
    fig.tight_layout()
    out = output_dir / "fig4_6_transfer.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"    wrote {out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--datasets", default="chestxray14,isic")
    parser.add_argument("--backbones", default="resnet18,dinov2_vits14")
    parser.add_argument(
        "--primary", default="chestxray14",
        help="Dataset used for the single-dataset figures",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Figure 4.1 - metric against distortion heat map")
    figure_heatmap(args.data_dir, args.output_dir, datasets)
    print("Figure 4.2 - paired bootstrap comparison")
    figure_paired(args.data_dir, args.output_dir, args.primary)
    print("Figure 4.3 - fusion strategies")
    figure_fusion(args.data_dir, args.output_dir, args.primary, backbones[0])
    print("Figure 4.4 - feature-set ablation")
    figure_ablation(args.data_dir, args.output_dir, datasets, backbones)
    print("Figure 4.5 - fitted coefficients")
    figure_coefficients(args.data_dir, args.output_dir, datasets, backbones[0])
    print("Figure 4.6 - cross-modality transfer")
    figure_transfer(args.data_dir, args.output_dir, datasets, backbones[0])


if __name__ == "__main__":
    main()