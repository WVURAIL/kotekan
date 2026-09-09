#!/usr/bin/env python3
"""Plot actual CPU accumulation receipts against independent analytic expectations."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASES = [
    "all-pairs-supported",
    "either-member-frame-rejected",
    "single-member-zero",
    "unequal-counts",
    "mixed-missing-and-rejected",
]
LABELS = [
    "All pairs\nsupported",
    "Two pairs\nrejected",
    "One-sided\nzero counts",
    "Unequal\ncounts",
    "Only one\nusable pair",
]


def read_receipts(directory):
    result = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        if record["schema"] != "n2-mask-normalization-cpu-v1":
            raise ValueError("unsupported receipt")
        key = (
            record["count_scale"],
            record["voltage_sample_period_fpga"],
            record["bin"],
            record["frequency_id"],
        )
        if key in result:
            raise ValueError("duplicate receipt identity")
        result[key] = record
    if not result:
        raise ValueError("empty receipt directory")
    return result


def main(args):
    for path in (args.pdf, args.png, args.summary):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    old, fixed = read_receipts(args.old_dir), read_receipts(args.fixed_dir)
    if len(old) != 48 or len(fixed) != 72:
        raise ValueError(
            "expected 48 original and 72 corrected full regression receipts"
        )
    for after in fixed.values():
        if not after["all_outputs_finite"]:
            raise ValueError("nonfinite corrected output")
        if not np.isclose(
            after["actual_weight"], after["expected_weight"], rtol=2e-5, atol=1e-7
        ):
            raise ValueError("corrected baseline weight does not agree")
        if not np.allclose(
            after["actual_visibility"],
            after["expected_visibility"],
            rtol=2e-6,
            atol=1e-7,
        ):
            raise ValueError("corrected baseline visibility does not agree")
        if any(
            after["actual_fpga_metadata"][k] != v
            for k, v in after["expected_fpga_metadata"].items()
        ):
            raise ValueError("corrected FPGA metadata does not agree")
    compared = []
    for key, before in old.items():
        after = fixed[key]
        for field in (
            "case",
            "admitted_sample_count",
            "usable_variance_pairs",
            "baseline",
        ):
            if before[field] != after[field]:
                raise ValueError(f"incomparable {field}")
        if not np.isclose(
            before["expected_weight"], after["expected_weight"], rtol=1e-12, atol=0
        ):
            raise ValueError("changed expected result")
        expected = after["expected_weight"]
        compared.append(
            {
                "identity": list(key),
                "case": after["case"],
                "expected_weight": expected,
                "old_actual_weight": before["actual_weight"],
                "fixed_actual_weight": after["actual_weight"],
                "old_ratio": before["actual_weight"] / expected
                if expected > 0
                else None,
                "fixed_ratio": after["actual_weight"] / expected
                if expected > 0
                else None,
            }
        )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    fig.text(
        0.075,
        0.95,
        "PATHFINDER IMPLEMENTATION / CPU REGRESSION",
        fontsize=9,
        color="#596876",
        weight="bold",
    )
    fig.text(
        0.075,
        0.90,
        "Correct precision after masking",
        fontsize=23,
        color="#202F3C",
        weight="bold",
    )
    fig.text(
        0.075,
        0.862,
        "Actual N2Accumulate outputs divided by independent analytic weights; 1 means agreement.",
        fontsize=11,
        color="#596876",
    )
    axes = fig.subplots(
        1,
        2,
        gridspec_kw={
            "left": 0.075,
            "right": 0.955,
            "top": 0.77,
            "bottom": 0.33,
            "wspace": 0.22,
        },
    )
    for ax, scale, title in zip(
        axes,
        (1, 4096),
        (
            "Up to 16 samples per subintegration",
            "Up to 65,536 samples: overflow regression",
        ),
    ):
        selected = []
        for case in CASES:
            matches = [
                r
                for r in compared
                if r["identity"][0] == scale
                and r["identity"][1] == 1
                and r["identity"][3] == 202
                and r["case"] == case
            ]
            if len(matches) != 1:
                raise ValueError("missing unique plotted case")
            selected.append(matches[0])
        x = np.arange(len(CASES))
        width = 0.34
        for offset, field, color, label in [
            (-width / 2, "old_ratio", "#EAAA00", "Archived pre-fix binary"),
            (width / 2, "fixed_ratio", "#005EB8", "Corrected CPU binary"),
        ]:
            values = [r[field] for r in selected]
            ax.bar(x + offset, values, width, color=color, label=label, zorder=3)
            for xx, value in zip(x + offset, values):
                ax.text(
                    xx,
                    value + 0.16,
                    f"{value:.0f}",
                    ha="center",
                    fontsize=9,
                    color="#202F3C",
                )
        ax.axhline(1, color="#202F3C", lw=1, ls="--", zorder=4)
        ax.set_ylim(0, 9)
        ax.set_xticks(x, LABELS, fontsize=9)
        ax.set_yticks([0, 1, 2, 4, 6, 8])
        ax.set_title(title, fontsize=11, weight="bold", loc="left", pad=15)
        ax.set_ylabel("Returned weight / expected weight")
        ax.grid(axis="y", alpha=0.18, zorder=0)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.text(
        0.075,
        0.248,
        r"$w = Nk/Q$; $k$ counts admitted pairs with two positive counts, not all nominal pairs.",
        fontsize=13,
        color="#202F3C",
    )
    fig.text(
        0.075,
        0.204,
        "Visibility means use all admitted surviving samples. A supported mean with no usable variance pair has weight 0.\n"
        "The pre-fix binary overstated precision by 2-4x in masked small-count cases; large counts exposed int32 overflow.\n"
        "Three zero-precision cases are retained in the tests but omitted from ratios because their expected weight is zero.",
        fontsize=10,
        color="#596876",
        linespacing=1.6,
        va="top",
    )
    fig.text(
        0.075,
        0.100,
        "Displayed: input pair (0, 2), frequency ID 202, fixed time bins. Full checks cover 64 inputs, 3 frequencies and 8 bins.\n"
        "Archived 868961b.dirty and corrected 997b15d.dirty are not a source-identical A/B build pair.\n"
        "Other fixed tests cover four-tick timing and cross-frame pairs. These fixtures do not calibrate physical noise.\n"
        "No GPU/telescope run, heterogeneous-baseline normalization or cleaner-transfer acceptance is implied.",
        fontsize=8.5,
        color="#596876",
        linespacing=1.5,
        va="top",
    )
    fig.savefig(
        args.pdf,
        metadata={
            "Title": "Pathfinder CPU accumulation normalization regression",
            "Subject": "Actual original/corrected CPU outputs; deterministic analytic fixture",
        },
    )
    fig.savefig(args.png, dpi=180)
    plt.close(fig)
    record = {
        "schema": "n2-normalization-comparison-v1",
        "comparisons": compared,
        "old_receipts": len(old),
        "fixed_receipts": len(fixed),
        "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "physical_noise_calibration": False,
        "source_identical_ab_build_pair": False,
    }
    args.summary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pdf": str(args.pdf), "comparisons": len(compared)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-dir", type=Path, required=True)
    p.add_argument("--fixed-dir", type=Path, required=True)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--png", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    main(p.parse_args())
