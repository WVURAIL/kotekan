#!/usr/bin/env python3
"""Plot the frozen CPU ensemble comparisons from the retained summary."""

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
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = json.loads((args.result / "summary.json").read_text())
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}",
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    cases = ["full", "independent-masks", "one-sided-mixed"]
    titles = ["Full support", "Independent masks", "One-sided and supported pairs"]
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 4.1), sharey=True)
    for ax, case, title in zip(axes, cases, titles):
        rows = [r for r in result["primary_rows"] if r["case"] == case]
        x = np.arange(len(rows))
        ax.axhspan(.85, 1.15, color="#e5eee8", label="Frozen engineering tolerance")
        ax.axhline(1, color="#333333", lw=.9)
        for shift, value, error, colour, marker, label in (
            (-.10, "empirical_variance_ratio", "empirical_ratio_mc_se", "#246085", "o", "Measured squared error"),
            (.10, "reported_variance_ratio", "reported_ratio_mc_se", "#a75727", "s", "Mean reciprocal precision"),
        ):
            ax.errorbar(x+shift, [r[value] for r in rows], yerr=[r[error] for r in rows],
                        fmt=marker, markersize=4, capsize=3, color=colour,
                        lw=1, label=label)
        ax.set_xticks(x, [f"({r['inputs'][0]}, {r['inputs'][1]})" for r in rows])
        ax.set_xlabel("Input product")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-.5, 3.5)
        ax.set_ylim(.80, 1.20)
        ax.set_yticks([.8,.9,1,1.1,1.2])
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Variance divided by exact model expectation")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles[1:]+handles[:1], labels[1:]+labels[:1], loc="lower center",
                  bbox_to_anchor=(.5,.065), ncol=3, frameon=False, fontsize=9)
    figure.suptitle("CPU accumulation: empirical and reported variance", y=.99, fontsize=14)
    figure.text(.5,.915,r"1,024 independent realizations per case; 16,384 samples per bin",
                ha="center", fontsize=10)
    figure.text(.5,.025,r"Whiskers: $\pm1$ estimated Monte Carlo standard error. Shading: fixed $\pm15\%$ tolerance, not confidence limits.",
                ha="center", fontsize=9)
    figure.subplots_adjust(left=.075,right=.99,bottom=.27,top=.81,wspace=.12)
    out=args.result/"figures"
    out.mkdir(exist_ok=True)
    stamp=datetime(2026,9,9,tzinfo=timezone.utc)
    figure.savefig(out/"n2-empirical-precision.pdf",metadata={"CreationDate":stamp,"ModDate":stamp})
    figure.savefig(out/"n2-empirical-precision.png",dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
