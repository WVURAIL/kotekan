#!/usr/bin/env python3
"""Plot retained composed-stage ensemble ratios without recomputing trials."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    result = json.loads((args.release / "summary.json").read_text())
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{lmodern}",
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.15), sharey=True)
    for ax, case, title in zip(
        axes,
        ("full-support", "heterogeneous-with-zero-fragments"),
        ("Full support", "Heterogeneous support and zero-count fragments"),
    ):
        rows = [r for r in result["primary_rows"] if r["case"] == case]
        x = np.arange(len(rows))
        ax.axhspan(0.85, 1.15, color="#e5eee8", label="Frozen engineering tolerance")
        ax.axhline(1, color="#303030", lw=0.9)
        for offset, key, colour, marker, label in (
            (
                -0.1,
                "empirical_variance_ratio",
                "#246085",
                "o",
                "Measured squared error",
            ),
            (
                0.1,
                "reported_variance_ratio",
                "#a75727",
                "s",
                "Mean reciprocal precision",
            ),
        ):
            ax.plot(
                x + offset,
                [np.nan if r[key] is None else r[key] for r in rows],
                marker=marker,
                linestyle="none",
                color=colour,
                markersize=4,
                lw=1,
                label=label,
            )
        ax.set_xticks(x, [f"({r['inputs'][0]}, {r['inputs'][1]})" for r in rows])
        ax.set_xlim(-0.5, 3.5)
        ax.set_xlabel("Input product")
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", alpha=0.2)
    extent = [
        r[k]
        for r in result["primary_rows"]
        for k in ("empirical_variance_ratio", "reported_variance_ratio")
        if r[k] is not None
    ]
    lower = min([0.8] + [v - 0.025 for v in extent])
    upper = max([1.2] + [v + 0.025 for v in extent])
    axes[0].set_ylim(lower, upper)
    axes[0].set_ylabel("Variance divided by ideal model expectation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[1:] + handles[:1],
        labels[1:] + labels[:1],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.064),
        ncol=3,
        frameon=False,
        fontsize=8.7,
    )
    fig.suptitle("Composed CPU stages: ensemble variance", fontsize=14, y=0.99)
    fig.text(
        0.5,
        0.91,
        r"1,024 groups per case; each combines four or five 16,384-sample source frames",
        ha="center",
        fontsize=9.6,
    )
    fig.text(
        0.5,
        0.022,
        r"Shading: fixed $\pm15\%$ engineering tolerance, not confidence limits.",
        ha="center",
        fontsize=8.8,
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.81, bottom=0.265, wspace=0.15)
    folder = args.release / "figures"
    folder.mkdir(exist_ok=True)
    stamp = datetime(2026, 9, 9, tzinfo=timezone.utc)
    fig.savefig(
        folder / "n2-composed-precision.pdf",
        metadata={"CreationDate": stamp, "ModDate": stamp},
    )
    fig.savefig(folder / "n2-composed-precision.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
