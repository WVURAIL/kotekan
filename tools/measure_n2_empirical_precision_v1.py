#!/usr/bin/env python3
"""Frozen independent-integer-voltage experiment against the CPU N2 stage.

Run ``freeze OUTPUT`` before ``run OUTPUT``. The latter never overwrites data.
Use ``smoke OUTPUT`` only for engineering setup with separate seeds/counts.
No telescope data, GPU stages, or distributional fitting is used.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E, S, SUBS, FRAME, BATCH = 64, 2048, 8, 16384, 128
ACTIVE = np.array([0, 1, 8, 16])
MU = np.array([0, 0, 1 + 1j, -1j])
AR, AC = np.triu_indices(len(ACTIVE))
ROWS, COLS = ACTIVE[AR], ACTIVE[AC]
PROBES = ((0, 0), (0, 1), (0, 8), (8, 16))
CASES = ("full", "independent-masks", "one-sided-mixed")
SOURCES = (
    "tools/measure_n2_empirical_precision_v1.py",
    "tests/test_n2_accumulate.py",
    "tests/test_n2_per_product_support.py",
    "python/kotekan/runner.py",
    "python/kotekan/chordbuffer.py",
    "python/kotekan/n2buffer.py",
    "python/kotekan/telescope.py",
    "lib/stages/N2Accumulate.cpp",
    "lib/stages/N2Accumulate.hpp",
    "build/pathfinder-normalization-cpu/kotekan/kotekan",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def product_moments():
    """Analytic finite-noise moments, complex E|z-Ez|² convention."""
    result = []
    for i, j in zip(AR, AC):
        if i == j:
            mean = abs(MU[i]) ** 2 + 4
            variance = 5.6 + 8 * abs(MU[i]) ** 2
        else:
            mean = MU[i] * MU[j].conjugate()
            variance = 16 + 4 * (abs(MU[i]) ** 2 + abs(MU[j]) ** 2)
        result.append(
            {
                "inputs": [int(ACTIVE[i]), int(ACTIVE[j])],
                "mean": [float(np.real(mean)), float(np.imag(mean))],
                "variance": float(variance),
                "variance_convention": "real variance"
                if i == j
                else "complex E|z-Ez|^2",
            }
        )
    return result


def design():
    return {
        "schema": "n2-empirical-precision-plan-v1",
        "status": "frozen before evaluation",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 202609090731,
        "generator": "numpy.PCG64; SeedSequence([seed, case_index, realization, stream]); stream0 voltages, stream1 masks",
        "realizations_per_case": 1024,
        "cases": list(CASES),
        "geometry": {
            "inputs": E,
            "active_inputs": ACTIVE.tolist(),
            "frequencies": [614],
            "samples_per_frame": FRAME,
            "subintegration_samples": S,
            "subintegrations_per_bin": SUBS,
            "voltage_sample_period_fpga_ticks": 1,
            "bin_in_ERA": False,
            "fringestop": False,
            "support_mode": "per_product_v1",
            "variance_mode": "EvenOddPosDef",
        },
        "voltage_model": {
            "component_support": [-2, -1, 0, 1, 2],
            "component_probabilities": [0.2] * 5,
            "independent_axes": [
                "realizations",
                "time",
                "active_inputs",
                "real_imaginary_components",
            ],
            "means": [[float(x.real), float(x.imag)] for x in MU],
            "other_inputs": "identically zero",
            "noise_complex_variance": 4,
            "noise_component_fourth_moment": 6.8,
        },
        "masks": {
            "full": "all samples, packet groups and frames admitted",
            "independent-masks": "packet presence is independent Bernoulli by sample and eight-input group with probabilities [0.92,0.72,0.55,0.85,0.80,0.75,0.70,0.65]; common fine keep is independent Bernoulli(0.9) per sample. Select exactly one of eight subintegrations uniformly for frame rejection, independent of voltages. Reject both members of that even/odd pair, retaining three pairs. No voltage/noise draw is replaced or selected.",
            "one-sided-mixed": "deterministic packet masks: group0 absent throughout subintegration1, group1 absent throughout subintegration3; all groups additionally present when (sample+3*group+subintegration) modulo (5+group modulo3) !=0. Common fine mask keeps sample when (sample+subintegration) modulo17 !=0. All frame gates admitted.",
            "guard": "separate 8 realizations: group0 present only in even subintegrations; other groups and common fine/frame gates all present. Probe products involving group0 have positive total support but K=0 and must report zero precision.",
            "independence": "all mask variables independent of voltage values; counts are exact intersections, never marginal approximations",
        },
        "moments": product_moments(),
        "primary_probes": [list(p) for p in PROBES],
        "estimands": {
            "conditional_variance": "sigma_z^2 / N for admitted individual sample products",
            "q": "sum_over_supported_admitted_pairs n0*n1/(n0+n1)*|mean0-mean1|^2",
            "reported_weight": "N*K/Q, or zero when N,K,Q do not support finite positive precision",
            "inverse_weight_target": "E[Q/(N*K) | masks]=sigma_z^2/N when K>0 under the stated independent common-variance/equal-mean law",
            "ratios": "sum |actual_mean-true_mean|^2 / sum conditional_variance; sum 1/actual_weight / sum conditional_variance. Full primary realization set; missing/zero precision is a failure, not dropped.",
            "not_target": "E[weight] is not reciprocal E[variance]; weight*error and nominal confidence coverage are not assumed calibrated",
        },
        "acceptance": {
            "ensemble_ratio_range": [0.85, 1.15],
            "interpretation": "fixed engineering tolerance, not a simultaneous confidence bound",
            "primary_comparisons": len(CASES) * len(PROBES) * 2,
            "arithmetic_mean_rtol": 2e-6,
            "arithmetic_mean_atol": 2e-7,
            "arithmetic_weight_rtol": 4e-6,
            "arithmetic_weight_atol": 2e-7,
            "counts": "exact for all 2080 products",
            "zero_precision": "retain and fail primary reciprocal comparison if any zero/nonfinite precision; guard K=0 requires exactly zero",
            "retry": "none; do not change seeds, counts, masks, gates or thresholds after inspection",
        },
        "source_sha256": source_hashes(),
        "numpy_version": np.__version__,
        "scope": "CPU accumulation and independent synthetic stationary integer-voltage model only; no physical precision, production mask independence, temporal-covariance or Pathfinder acceptance claim",
    }


def generate(seed, case, realization):
    """Return unmasked integer voltage components, group masks, and frame gates."""
    vrng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, case, realization, 0]))
    )
    mrng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, case, realization, 1]))
    )
    components = vrng.integers(-2, 3, (SUBS, S, len(ACTIVE), 2), dtype=np.int8)
    components[..., 0] += MU.real.astype(np.int8)
    components[..., 1] += MU.imag.astype(np.int8)
    present = np.ones((SUBS, S, E // 8), dtype=bool)
    keep = np.ones((SUBS, S), dtype=bool)
    admit = np.ones(SUBS, dtype=np.uint8)
    if case == 1:
        present = mrng.random(present.shape) < np.array(
            [0.92, 0.72, 0.55, 0.85, 0.80, 0.75, 0.70, 0.65]
        )
        keep = mrng.random(keep.shape) < 0.9
        admit[int(mrng.integers(SUBS))] = 0
    elif case == 2:
        t, sample, group = np.indices(present.shape)
        present = (sample + 3 * group + t) % (5 + group % 3) != 0
        present[1, :, 0] = False
        present[3, :, 1] = False
        keep = (np.arange(S)[None, :] + np.arange(SUBS)[:, None]) % 17 != 0
    elif case == 3:
        present[1::2, :, 0] = False
    return components, present, keep, admit


def sufficient(components, present, keep, admit):
    valid = present & keep[..., None]
    x = components[..., 0].astype(float) + 1j * components[..., 1]
    x *= valid[..., ACTIVE // 8]
    raw = np.empty((SUBS, len(AR)), dtype=np.complex128)
    joint = np.empty((SUBS, E // 8, E // 8), dtype=np.int64)
    for t in range(SUBS):
        raw[t] = (x[t].T @ x[t].conjugate())[AR, AC]
        joint[t] = valid[t].astype(np.int64).T @ valid[t].astype(np.int64)
    counts = joint[:, ROWS // 8, COLS // 8]
    pair_admit = admit[0::2].astype(bool) & admit[1::2].astype(bool)
    use = np.repeat(pair_admit, 2)
    n = counts[use].sum(axis=0)
    total = raw[use].sum(axis=0)
    k, q = np.zeros(len(AR), dtype=int), np.zeros(len(AR))
    for t in range(0, SUBS, 2):
        supported = use[t] & (counts[t] > 0) & (counts[t + 1] > 0)
        k += supported
        n0, n1 = counts[t, supported], counts[t + 1, supported]
        delta = raw[t, supported] / n0 - raw[t + 1, supported] / n1
        q[supported] += (n0 * n1 / (n0 + n1)) * np.abs(delta) ** 2
    mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
    weight = np.divide(n * k, q, out=np.zeros_like(q), where=(q > 0) & (k > 0))
    return raw, joint, n, k, q, mean, weight, use


def native_streams(number):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_n2_accumulate import make_zeroed_chord_buffer

    def buf(name, dtype, typename, tail=(), dims=(), scales=(), extra=None):
        return make_zeroed_chord_buffer(
            name,
            dtype,
            typename,
            (SUBS, 1) + tail,
            ("Tc", "F") + dims,
            (S, 1) + scales,
            FRAME,
            FRAME,
            number,
            freq_ids=np.array([614], dtype=np.int32),
            time_downsampling=S,
            extra_meta=extra,
        )

    return {
        "in_buf": buf(
            "n2k_correlation",
            np.int32,
            "int32",
            (10, 16, 16, 2),
            ("DPhi", "DPlo1", "DPlo2", "C"),
            (16, 1, 1, 1),
        ),
        "in_counts_buf": buf(
            "n2k_counts",
            np.int32,
            "int32",
            (1, 8, 8),
            ("D8Phi", "D8Plo1", "D8Plo2"),
            (64, 8, 8),
        ),
        "in_rficounts_buf": buf("RFImask_counts", np.int32, "int32"),
        "in_plcounts_buf": buf("pl_lost_counts_scalar", np.int32, "int32"),
        "in_rfiframemask_buf": buf(
            "RFIFrameMask",
            np.uint8,
            "uint8",
            extra={
                "rfi_frame_excision_enabled": True,
                "rfi_frame_excision_thresholds": np.array(
                    [[3.0, 0.1]], dtype=np.float32
                ),
            },
        ),
    }


def run_batch(destination, seed, case, start, number):
    from copy import deepcopy

    from test_n2_accumulate import chime_tel
    from test_n2_per_product_support import PerProductDump

    from kotekan import runner

    streams = native_streams(number)
    truth = []
    active_block = (COLS // 16) * (COLS // 16 + 1) // 2 + ROWS // 16
    full_rows, full_cols = np.triu_indices(E)
    probe_indices = [
        int(np.flatnonzero((ROWS == i) & (COLS == j))[0]) for i, j in PROBES
    ]
    n2indices = np.array(
        [
            int(np.flatnonzero((full_rows == i) & (full_cols == j))[0])
            for i, j in zip(ROWS, COLS)
        ]
    )
    zero_indices = np.setdiff1d(np.arange(len(full_rows)), n2indices)
    voltage_hash = []
    for b in range(number):
        comp, present, keep, admit = generate(seed, case, start + b)
        raw, joint, n, k, q, mean, weight, use = sufficient(comp, present, keep, admit)
        voltage_hash.append(hashlib.sha256(comp.tobytes()).hexdigest())
        for t in range(SUBS):
            co = streams["in_buf"][b].data[t, 0]
            co[active_block, COLS % 16, ROWS % 16, 0] = raw[t].real
            co[active_block, COLS % 16, ROWS % 16, 1] = -raw[t].imag
            streams["in_counts_buf"][b].data[t, 0, 0] = joint[t]
            streams["in_rficounts_buf"][b].data[t, 0] = (~keep[t]).sum()
            streams["in_plcounts_buf"][b].data[t, 0] = (~present[t, :, 0]).sum()
            streams["in_rfiframemask_buf"][b].data[t, 0] = admit[t]
        truth.append((raw, joint, n, k, q, mean, weight, use, admit))
    with tempfile.TemporaryDirectory(prefix="n2-empirical-") as tmp:
        readers = {
            key: runner.ReadChordBuffer(tmp, frames) for key, frames in streams.items()
        }
        for reader in readers.values():
            reader.write()
        output = PerProductDump(
            tmp, exit_after_n_files=number, num_elements=E, num_ev=0, num_freq=1
        )
        config = {
            "buffer_depth": 3,
            "samples_per_data_set": FRAME,
            "sub_integration_ntime": S,
            "num_local_freq": 1,
            "num_elements": E,
            "num_polarizations": 2,
            "num_dishes": E // 2,
            "num_ev": 0,
            "telescope": deepcopy(chime_tel),
        }
        boot = 1767287750500000000
        config["telescope"]["frame0_nano"] = boot
        config["gps_time"] = {"frame0_nano": boot}
        stage = runner.KotekanStageTester(
            "N2Accumulate",
            {
                "num_freq_per_n2k_frame": 1,
                "packet_loss_is_scalar": False,
                "bin_in_ERA": False,
                "num_subintegrations_per_bin": SUBS,
                "variance_mode": "EvenOddPosDef",
                "do_fringestop": False,
                "input_order": "CHIMEBeamformer",
            },
            readers,
            output,
            config,
        )
        capture = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(capture),
                contextlib.redirect_stderr(capture),
            ):
                stage.run()
        finally:
            (destination / "stage.log").write_text(capture.getvalue())
            # Preserve actual serialized inputs and native output frames for an
            # independent decoder, including on failure. Compression is lossless.
            with zipfile.ZipFile(
                destination / "native-io.zip",
                "w",
                zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as z:
                for p in sorted(Path(tmp).rglob("*")):
                    if p.is_file():
                        z.write(p, p.relative_to(tmp).as_posix())
        actual = sorted(output.load(), key=lambda f: int(f.metadata.abs_time_idx))
        assert len(actual) == number
        records = {
            key: []
            for key in (
                "actual_mean",
                "actual_weight",
                "actual_count",
                "n",
                "k",
                "q",
                "raw",
                "counts",
                "admit",
                "full_count_error",
                "zero_payload_errors",
                "metadata_errors",
            )
        }
        checks, errors = 0, []
        for b, (got, ref) in enumerate(zip(actual, truth)):
            raw, joint, n, k, q, mean, weight, use, admit = ref
            all_counts = joint[use].sum(axis=0)[full_rows // 8, full_cols // 8]
            count_errors = int(np.count_nonzero(got.valid_fpga_ticks != all_counts))
            zero_errors = int(np.count_nonzero(got.vis[zero_indices])) + int(
                np.count_nonzero(got.weight[zero_indices])
            )
            metadata_errors = (
                int(got.metadata.frame_length_fpga_ticks != FRAME)
                + int(got.metadata.fpga_start_tick != (b + 1) * FRAME)
                + int(got.metadata.freq_id != 614)
            )
            checks += len(full_rows) + len(zero_indices) * 2 + 3 + len(AR) * 2
            if count_errors or zero_errors or metadata_errors:
                errors.append(
                    [
                        b,
                        "support/zero/metadata",
                        count_errors,
                        zero_errors,
                        metadata_errors,
                    ]
                )
            if not np.allclose(got.vis[n2indices], mean, rtol=2e-6, atol=2e-7):
                errors.append([b, "mean"])
            if not np.allclose(got.weight[n2indices], weight, rtol=4e-6, atol=2e-7):
                errors.append([b, "weight"])
            values = {
                "actual_mean": got.vis[n2indices].copy(),
                "actual_weight": got.weight[n2indices].copy(),
                "actual_count": got.valid_fpga_ticks[n2indices].copy(),
                "n": n,
                "k": k,
                "q": q,
                "raw": raw,
                "counts": joint[:, ROWS // 8, COLS // 8],
                "admit": admit,
                "full_count_error": count_errors,
                "zero_payload_errors": zero_errors,
                "metadata_errors": metadata_errors,
            }
            for key, value in values.items():
                records[key].append(value)
        np.savez_compressed(
            destination / "receipts.npz",
            **{key: np.asarray(v) for key, v in records.items()},
            voltage_sha256=np.array(voltage_hash),
            realization=np.arange(start, start + number),
            probe_indices=probe_indices,
        )
        write_json(
            destination / "arithmetic.json",
            {
                "checks": checks,
                "failures": errors,
                "case": case,
                "start": start,
                "number": number,
            },
        )
        return checks, len(errors)


def summarize(out, plan):
    moments = product_moments()
    means = np.array([complex(*x["mean"]) for x in moments])
    variances = np.array([x["variance"] for x in moments])
    pidx = [i for i, m in enumerate(moments) if tuple(m["inputs"]) in PROBES]
    rows = []
    for case in CASES:
        files = sorted((out / "batches" / case).glob("*/receipts.npz"))
        arrays = {
            k: np.concatenate([np.load(p)[k] for p in files])
            for k in ("actual_mean", "actual_weight", "actual_count", "k")
        }
        for p in pidx:
            count = arrays["actual_count"][:, p].astype(float)
            weight = arrays["actual_weight"][:, p].astype(float)
            good = (count > 0) & (weight > 0) & np.isfinite(weight)
            target = np.divide(
                variances[p], count, out=np.full_like(count, np.nan), where=count > 0
            )
            mse = np.abs(arrays["actual_mean"][:, p] - means[p]) ** 2
            inv = np.divide(1, weight, out=np.full_like(weight, np.nan), where=good)

            def ratio(x, target=target):
                return (
                    float(np.sum(x) / np.sum(target))
                    if np.all(np.isfinite(x)) and np.all(np.isfinite(target))
                    else None
                )

            rmean, rinv = ratio(mse), ratio(inv)
            rows.append(
                {
                    "case": case,
                    "inputs": moments[p]["inputs"],
                    "realizations": len(count),
                    "zero_or_nonfinite_precision": int((~good).sum()),
                    "sample_count_min": int(count.min()),
                    "sample_count_max": int(count.max()),
                    "supported_pairs_min": int(arrays["k"][:, p].min()),
                    "supported_pairs_max": int(arrays["k"][:, p].max()),
                    "theoretical_mean": moments[p]["mean"],
                    "single_sample_variance": variances[p],
                    "mean_conditional_variance": float(np.nanmean(target)),
                    "mean_squared_error": float(mse.mean()),
                    "mean_inverse_weight": float(inv.mean()) if np.all(good) else None,
                    "empirical_variance_ratio": rmean,
                    "reported_variance_ratio": rinv,
                    "empirical_ratio_mc_se": float(
                        np.std(mse - target, ddof=1)
                        / math.sqrt(len(count))
                        / np.mean(target)
                    )
                    if np.all(np.isfinite(target))
                    else None,
                    "reported_ratio_mc_se": float(
                        np.std(inv - target, ddof=1)
                        / math.sqrt(len(count))
                        / np.mean(target)
                    )
                    if np.all(good)
                    else None,
                    "passes_engineering_tolerance": all(
                        r is not None and 0.85 <= r <= 1.15 for r in (rmean, rinv)
                    ),
                }
            )
    guard_files = sorted((out / "batches" / "guard").glob("*/receipts.npz"))
    guard = {
        k: np.concatenate([np.load(p)[k] for p in guard_files])
        for k in ("actual_weight", "actual_count", "k")
    }
    gp = [p for p in pidx if moments[p]["inputs"][0] in (0, 1)]
    guard_pass = bool(
        np.all(guard["actual_weight"][:, gp] == 0)
        and np.all(guard["actual_count"][:, gp] > 0)
        and np.all(guard["k"][:, gp] == 0)
    )
    arithmetic = [
        json.loads(p.read_text()) for p in (out / "batches").rglob("arithmetic.json")
    ]
    result = {
        "schema": "n2-empirical-precision-summary-v1",
        "plan_sha256": sha(out / "plan.json"),
        "geometry": plan["geometry"],
        "primary_comparisons": len(rows) * 2,
        "primary_rows": rows,
        "arithmetic_checks": sum(x["checks"] for x in arithmetic),
        "arithmetic_failed_batches": sum(bool(x["failures"]) for x in arithmetic),
        "guard_realizations": len(guard["k"]),
        "guard_probe_checks": len(guard["k"]) * len(gp),
        "guard_pass": guard_pass,
        "all_engineering_checks_pass": all(
            r["passes_engineering_tolerance"] for r in rows
        )
        and guard_pass
        and not any(x["failures"] for x in arithmetic),
        "scope": plan["scope"],
        "acceptance_interpretation": plan["acceptance"]["interpretation"],
        "mc_se_interpretation": "descriptive estimated Monte Carlo SE across independent realizations; no confidence-level or Gaussian-distribution claim",
    }
    write_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run", "smoke"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    if args.command in ("freeze", "smoke"):
        out.mkdir(parents=True, exist_ok=False)
        plan = design()
        if args.command == "smoke":
            plan["status"] = "engineering smoke; not scientific evaluation"
            plan["seed"] = 202609090000
            plan["realizations_per_case"] = 4
        write_json(out / "plan.json", plan)
        (out / "plan.sha256").write_text(sha(out / "plan.json") + "\n")
        if args.command == "freeze":
            return
    else:
        plan = json.loads((out / "plan.json").read_text())
    assert sha(out / "plan.json") == (out / "plan.sha256").read_text().strip()
    assert plan["source_sha256"] == source_hashes(), "Frozen source/binary changed"
    assert np.__version__ == plan["numpy_version"], "Frozen numpy version changed"
    (out / "batches").mkdir(exist_ok=False)
    sys.path.insert(0, str(ROOT / "tests"))
    import time

    started = time.monotonic()
    for ci, case in enumerate((*CASES, "guard")):
        total = plan["realizations_per_case"] if ci < 3 else 8
        for start in range(0, total, BATCH):
            destination = out / "batches" / case / f"{start:05d}"
            destination.mkdir(parents=True)
            checks, failures = run_batch(
                destination, plan["seed"], ci, start, min(BATCH, total - start)
            )
            print(
                json.dumps(
                    {
                        "case": case,
                        "done": start + min(BATCH, total - start),
                        "total": total,
                        "checks": checks,
                        "failed_arithmetic_rows": failures,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    result = summarize(out, plan)
    write_json(
        out / "source-integrity.json",
        {
            "unchanged": plan["source_sha256"] == source_hashes(),
            "sha256": source_hashes(),
        },
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
