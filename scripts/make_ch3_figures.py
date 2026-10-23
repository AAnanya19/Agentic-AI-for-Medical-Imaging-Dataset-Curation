#!/usr/bin/env python3
"""Generate the figures for Chapter 3 from the project's own log files.

Writes publication-quality figures to ``figures/`` for insertion into the
dissertation. Each figure is generated independently and a missing input file
causes that figure to be skipped with an explanation rather than aborting the
run, so a partial set of logs still produces whatever it can support.

    python scripts/make_ch3_figures.py --data-dir data --output-dir figures

Figures produced:

    fig3_1_resolution_distribution   source resolution spread, both datasets
    fig3_3_calibrated_parameters     calibrated parameter by modality
    fig3_5_severity_validation       median SSIM by severity, full corpus

Figures 3.2 and 3.4 are the defect example grids already produced by
``inject_defects.py --visualize``; see the note printed at the end of this
script for which files to use.
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

# Greyscale-safe, print-legible defaults. Figures are read in a bound document,
# often photocopied, so colour alone must never carry meaning.
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

DATASET_LABELS = {"chestxray14": "ChestX-ray14", "isic": "ISIC"}
# Distinguished by hatch as well as shade so the figures survive greyscale
# printing.
DATASET_STYLE = {
    "chestxray14": {"color": "#4C72B0", "hatch": ""},
    "isic": {"color": "#DD8452", "hatch": "///"},
}
SEVERITY_TARGETS = {1: 0.963, 2: 0.846, 3: 0.729}


def tidy(name: str) -> str:
    return name.replace("_", " ")


# ---------------------------------------------------------------- figure 3.1
def figure_resolution(data_dir: Path, output_dir: Path, datasets: list[str]) -> None:
    """Source resolution distribution, showing the disparity that motivates
    canonical preprocessing."""
    frames: dict[str, pd.Series] = {}
    for dataset in datasets:
        path = data_dir / dataset / "processed_metadata.csv"
        if not path.is_file():
            print(f"  skip {dataset}: {path} not found")
            continue
        processed = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "defect_params_json" not in processed.columns:
            print(f"  skip {dataset}: no defect_params_json column")
            continue
        megapixels = []
        for raw in processed["defect_params_json"]:
            try:
                info = json.loads(raw)["baseline_preprocessing"]["canonical_resize"]
                megapixels.append(
                    float(info["original_width"]) * float(info["original_height"]) / 1e6
                )
            except Exception:  # noqa: BLE001 - a malformed row is simply skipped
                continue
        if megapixels:
            frames[dataset] = pd.Series(megapixels)

    if not frames:
        print("  figure 3.1 skipped: no resolution data available")
        return

    # A SHARED x-axis is essential: the comparison between a uniform and a
    # four-order-of-magnitude distribution is the entire point of the figure, and
    # independent axes would hide it by rescaling each panel to fill its own
    # width.
    all_positive = pd.concat([v[v > 0] for v in frames.values()])
    bins = np.logspace(
        np.log10(all_positive.min() * 0.8), np.log10(all_positive.max() * 1.2), 45
    )

    fig, axes = plt.subplots(
        len(frames), 1, figsize=(6.4, 3.6), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, (dataset, values) in zip(axes, frames.items()):
        style = DATASET_STYLE.get(dataset, {"color": "#555555", "hatch": ""})
        positive = values[values > 0]
        axis.hist(positive, bins=bins, color=style["color"], hatch=style["hatch"],
                  edgecolor="black", linewidth=0.4)
        axis.set_xscale("log")
        axis.set_ylabel("images")
        span = (
            "uniform" if positive.nunique() == 1
            else f"{positive.min():.3g} to {positive.max():.3g} MP"
        )
        axis.set_title(
            f"{DATASET_LABELS.get(dataset, dataset)}: "
            f"median {positive.median():.2f} MP, {span}",
            loc="left",
        )
    axes[-1].set_xlabel("source resolution (megapixels, log scale)")
    fig.tight_layout()
    path = output_dir / "fig3_1_resolution_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- figure 3.3
def figure_calibration(data_dir: Path, output_dir: Path, datasets: list[str]) -> None:
    """Calibrated parameter per modality, normalised so distortions with
    incomparable units can share an axis."""
    records: list[dict] = []
    for dataset in datasets:
        path = data_dir / "logs" / f"{dataset}_severity_calibration.json"
        if not path.is_file():
            print(f"  skip {dataset}: {path} not found")
            continue
        payload = json.loads(path.read_text())
        entries = payload.get("results") or payload.get("calibration") or payload
        if isinstance(entries, dict):
            for defect, by_severity in entries.items():
                if not isinstance(by_severity, dict):
                    continue
                for severity, entry in by_severity.items():
                    if not isinstance(entry, dict):
                        continue
                    value = entry.get("value", entry.get("parameter"))
                    if value is None:
                        continue
                    try:
                        records.append({
                            "dataset": dataset, "defect": defect,
                            "severity": int(str(severity)[-1]),
                            "value": float(value),
                        })
                    except (TypeError, ValueError):
                        continue

    if not records:
        print(
            "  figure 3.3 skipped: could not parse calibration JSON. Inspect "
            "data/logs/<dataset>_severity_calibration.json and adjust the keys "
            "read in figure_calibration()."
        )
        return

    frame = pd.DataFrame(records)
    severe = frame[frame["severity"] == 3]
    if severe.empty:
        print("  figure 3.3 skipped: no severity-3 entries found")
        return

    pivot = severe.pivot_table(
        index="defect", columns="dataset", values="value", aggfunc="first"
    ).dropna()
    if pivot.empty or pivot.shape[1] < 2:
        print("  figure 3.3 skipped: need both datasets to compare")
        return

    # Ratio rather than raw value: sigma, kernel length and scale factor are not
    # on comparable scales, but their ratio between modalities is.
    first, second = pivot.columns[:2]
    ratio = (pivot[second] / pivot[first]).sort_values()

    fig, axis = plt.subplots(figsize=(6.2, 3.0))
    positions = np.arange(len(ratio))
    axis.barh(positions, ratio.values, color="#4C72B0", edgecolor="black",
              linewidth=0.4)
    axis.axvline(1.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels([tidy(name) for name in ratio.index])
    axis.set_xlabel(
        f"parameter ratio, {DATASET_LABELS.get(second, second)} relative to "
        f"{DATASET_LABELS.get(first, first)}"
    )
    axis.set_title("Calibrated parameter required for equal structural degradation")
    for position, value in zip(positions, ratio.values):
        axis.text(value + 0.05, position, f"{value:.1f}x", va="center", fontsize=8)
    axis.set_xlim(0, max(ratio.values) * 1.25)
    fig.tight_layout()
    path = output_dir / "fig3_3_calibrated_parameters.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- figure 3.5
def figure_validation(data_dir: Path, output_dir: Path, datasets: list[str]) -> None:
    """Median SSIM by severity for every distortion, measured on the full corpus."""
    frames: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        path = data_dir / "logs" / f"{dataset}_ground_truth_summary.csv"
        if not path.is_file():
            print(f"  skip {dataset}: {path} not found")
            continue
        summary = pd.read_csv(path)
        if not {"defect_type", "severity_level", "median"} <= set(summary.columns):
            print(f"  skip {dataset}: unexpected columns in {path.name}")
            continue
        frames[dataset] = summary

    if not frames:
        print("  figure 3.5 skipped: no ground truth summaries found")
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(7.2, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (dataset, summary) in zip(axes, frames.items()):
        summary = summary.copy()
        summary["severity_level"] = pd.to_numeric(
            summary["severity_level"], errors="coerce"
        )
        for target in SEVERITY_TARGETS.values():
            axis.axhline(target, color="black", linewidth=0.7, linestyle=":",
                         zorder=1)
        for defect, group in summary.groupby("defect_type"):
            ordered = group.sort_values("severity_level")
            # Rotation is plotted dashed because its severity is not ordinal
            # under this metric, as established in Section 3.6.
            is_rotation = "rotation" in str(defect)
            axis.plot(
                ordered["severity_level"], ordered["median"],
                marker="o", markersize=3, linewidth=1.0,
                linestyle="--" if is_rotation else "-",
                color="black" if is_rotation else None,
                label=tidy(str(defect)), zorder=3,
            )
        axis.set_xticks([1, 2, 3])
        axis.set_xlabel("injected severity")
        axis.set_title(DATASET_LABELS.get(dataset, dataset))
    axes[0].set_ylabel("median structural similarity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False)
    fig.tight_layout()
    path = output_dir / "fig3_5_severity_validation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--datasets", default="chestxray14,isic")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Figure 3.1 - source resolution distribution")
    figure_resolution(args.data_dir, args.output_dir, datasets)
    print("Figure 3.3 - calibrated parameters by modality")
    figure_calibration(args.data_dir, args.output_dir, datasets)
    print("Figure 3.5 - severity validation across the full corpus")
    figure_validation(args.data_dir, args.output_dir, datasets)

    print()
    print("Figures 3.2 and 3.4 use the defect example grids already produced by")
    print("inject_defects.py --visualize:")
    print("  Figure 3.2: data/logs/defect_examples_chestxray14.png")
    print("  Figure 3.4: data/logs/defect_examples_isic.png, before and after the")
    print("              blur cap. If the pre-cap version was not retained, note")
    print("              in the caption that only the constrained version is shown.")


if __name__ == "__main__":
    main()