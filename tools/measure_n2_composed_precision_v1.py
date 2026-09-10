#!/usr/bin/env python3
"""Frozen synthetic N2Accumulate output replay into actual N2TimeDownsample."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import time
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
import measure_n2_empirical_precision_v1 as base
from test_n2_accumulate import chime_tel
from test_n2_per_product_support import PerProductDump
from test_n2_time_downsample_normalization import BOOT, TELESCOPE

from kotekan import runner
from kotekan import telescope as tel

FRAME, SUBS, S, E = 16384, 8, 2048, 64
GROUPS_PER_BATCH, BATCHES, ERA_BINS = 16, 64, 500000
CASE_NAMES = (
    "full-support",
    "heterogeneous-with-zero-fragments",
    "unavailable-precision-guard",
)
FULL_R, FULL_C = np.triu_indices(E)
PIND = np.array(
    [
        np.flatnonzero((FULL_R == i) & (FULL_C == j))[0]
        for i, j in zip(base.ROWS, base.COLS)
    ]
)
PROBES = [
    i for i, (a, b) in enumerate(zip(base.ROWS, base.COLS)) if (a, b) in base.PROBES
]
SOURCES = tuple(
    dict.fromkeys(
        (
            *base.SOURCES,
            "tools/measure_n2_composed_precision_v1.py",
            "tests/test_n2_time_downsample_normalization.py",
            "lib/stages/N2TimeDownsample.cpp",
            "lib/stages/N2TimeDownsample.hpp",
        )
    )
)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def hashes():
    return {p: base.sha(ROOT / p) for p in SOURCES}


def timeline():
    """Freeze all source identities and groups using time only, before voltages."""
    seq = (np.arange(128, dtype=np.int64) + 1) * FRAME
    centers = tel.get_t_inst_ns(seq + FRAME // 2, TELESCOPE)
    eops = tel.get_EOP_at_t_inst_ns(centers, TELESCOPE, False)
    rotations = tel.get_nrot_at_t_inst_ns(centers, TELESCOPE, False)
    boot = tel.get_t_inst_ns(0, TELESCOPE)
    initial_eop = tel.get_EOP_at_t_inst_ns(boot, TELESCOPE, False)
    initial_rotation = int(tel.get_nrot_at_t_inst_ns(boot, TELESCOPE, False))
    initial_bin = int(initial_eop.ERA_deg * ERA_BINS / 360)
    keys = [
        int(rotations[i] - initial_rotation) * ERA_BINS
        + int(eops[i].ERA_deg * ERA_BINS / 360)
        - initial_bin
        for i in range(len(seq))
    ]
    unique = list(dict.fromkeys(keys))
    stop = keys.index(unique[GROUPS_PER_BATCH + 1]) + 1
    keys = keys[:stop]
    groups = [
        {
            "relative_era_bin": key,
            "source_indices": [i for i, k in enumerate(keys) if k == key],
        }
        for key in unique[1 : GROUPS_PER_BATCH + 1]
    ]
    return {
        "source_frames_per_batch": stop,
        "source_start_ticks": seq[:stop].tolist(),
        "source_bin_keys": keys,
        "complete_groups": groups,
        "dropped_initial_source_indices": [
            i for i, k in enumerate(keys) if k == keys[0]
        ],
        "final_flush_trigger_source_index": stop - 1,
        "boundary_rule": "All frames in first observed ERA group skipped for stage alignment; first frame of the group after the16th complete group flushes that last complete group and is not emitted; exact indices are fixed from time before voltage generation.",
    }


def design():
    return {
        "schema": "n2-composed-precision-plan-v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 202609091629,
        "batches_per_primary_case": BATCHES,
        "guard_batches": 1,
        "primary_cases": list(CASE_NAMES[:2]),
        "guard_case": CASE_NAMES[2],
        "output_groups_per_batch": GROUPS_PER_BATCH,
        "primary_output_groups_per_case": BATCHES * GROUPS_PER_BATCH,
        "geometry": {
            "inputs": E,
            "active_inputs": base.ACTIVE.tolist(),
            "source_frame_samples": FRAME,
            "source_subintegration_samples": S,
            "source_subintegrations_per_bin": SUBS,
            "frequencies": [614],
            "support_mode": "per_product_v1",
            "num_bins_per_rotation": ERA_BINS,
            "accumulator_bin_in_ERA": False,
            "fringestop": False,
            "voltage_sample_period_fpga_ticks": 1,
            "telescope": TELESCOPE,
        },
        "timeline": timeline(),
        "law": "Independent uniform integer real/imaginary components {-2,-1,0,1,2}, plus means [0,0,1+i,-i] on active inputs [0,1,8,16]; other60 inputs exactly zero. Independent draws across source frames, time and active inputs. Same law as the imported finite-moment oracle; fresh seed/identities.",
        "generator": "base.generate(seed, mask_kind, identity), with identity=study_case*1000000+batch*1000+source_index; mask_kind is0(full),1(independent packet/fine plus one rejected frame),2(one-sided mixed),3(group0 only even subintegrations). Seed/identity streams are disjoint from prior releases.",
        "masks": {
            "full-support": "mask_kind0 for every source frame",
            "heterogeneous-with-zero-fragments": "Use mask_kind2 for member1 of each complete output group and mask_kind1 otherwise. For member0 of even-numbered complete groups, all packet groups absent; for member0 of odd-numbered groups, packet group0 absent while others keep the independent masks. All other members unchanged. Boundary source frames use mask_kind1. Masks independent of voltages; no draw replaced or selected.",
            "unavailable-precision-guard": "Use mask_kind3 for member1 in each complete group, full support otherwise. This creates positive-count zero-precision source products involving group0; downstream must retain their mean/count and emit zero precision. Cross product(8,16) remains a supported positive-precision control.",
        },
        "moments": base.product_moments(),
        "primary_probes": [list(x) for x in base.PROBES],
        "estimand": {
            "N": "sum of actual per-product admitted source counts N_i within the frozen output group",
            "ideal_conditional_variance": "sigma_z^2/N for the ideal unrounded raw-sample mean; actual float32 means/precision are separately compared with their exact arithmetic references within fixed numerical tolerances",
            "propagated_inverse_precision": "sum(N_i^2/w_i)/N^2 over positive-count source frames, only when every such w_i is finite and positive; source zero counts are ignored",
            "unbiased_ideal_estimator": "sum(N_i^2*Q_i/(N_i*K_i))/N^2 has conditional expectation sigma_z^2/N for independent source samples with common variance/equal expected visibility and K_i>0. This does not require mean/variance-estimator independence. Q_i=0 is included in this ideal identity, but actual zero precision has no reciprocal and fails primary qualification without exclusion.",
            "primary_ratios": "sum squared final-mean errors about known population mean divided by sum sigma_z^2/N; sum reciprocal final precision divided by same denominator. All1024 predetermined complete groups retained per case; any unavailable positive-count upstream precision fails its primary row even if downstream outputs a finite value.",
            "not_claimed": "E[precision]=inverse E[variance], calibrated confidence coverage, exact unbiasedness after float32 rounding, physical variance or operational live chain qualification",
        },
        "acceptance": {
            "ensemble_range": [0.85, 1.15],
            "primary_comparisons": 16,
            "meaning": "predeclared engineering tolerance, not simultaneous confidence limits",
            "mean_rtol": 2e-6,
            "mean_atol": 3e-7,
            "weight_rtol": 5e-6,
            "weight_atol": 3e-7,
            "counts": "exact for all2080 products at both stages",
            "active_arithmetic": "all10 active products at both stages, native-output replay bytes unchanged",
            "guard": "all predetermined positive-count unknown-precision outputs must have weight exactly0 and valid mean/count; supported control remains positive; no selection/retries",
        },
        "execution": "actual CPU N2Accumulate output frames serialized unchanged and replayed through actual CPU N2TimeDownsample in a second stage invocation; not jointly scheduled live pipeline",
        "source_sha256": hashes(),
        "numpy_version": np.__version__,
    }


def freeze(out, smoke=False):
    out.mkdir(parents=True, exist_ok=False)
    plan = design()
    if smoke:
        plan["seed"] = 202609091600
        plan["batches_per_primary_case"] = 1
        plan["primary_output_groups_per_case"] = GROUPS_PER_BATCH
        plan["scope"] = (
            "separate-seed engineering smoke only; not primary scientific evaluation"
        )
    save(out / "plan.json", plan)
    (out / "plan.sha256").write_text(base.sha(out / "plan.json") + "\n")
    for name in plan["source_sha256"]:
        dest = out / "source_snapshots" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)


def source_recipe(plan, case, batch, source_index):
    group_member = {
        index: (g, j)
        for g, group in enumerate(plan["timeline"]["complete_groups"])
        for j, index in enumerate(group["source_indices"])
    }
    group, within = group_member.get(source_index, (-1, -1))
    kind = (
        0
        if case == 0
        else (2 if within == 1 else 1)
        if case == 1
        else (3 if within == 1 else 0)
    )
    identity = case * 1000000 + batch * 1000 + source_index
    component, packet, fine, admit = base.generate(plan["seed"], kind, identity)
    if case == 1 and within == 0:
        if group % 2 == 0:
            packet[:] = False
        else:
            packet[:, :, 0] = False
    return component, packet, fine, admit, kind, identity


def stage_run(stage, tmp, archive, logfile):
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            stage.run()
    finally:
        logfile.write_text(capture.getvalue())
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(Path(tmp).rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(tmp).as_posix())


def run_batch(out, plan, case, batch):
    timeline = plan["timeline"]
    number = timeline["source_frames_per_batch"]
    streams = base.native_streams(number)
    reference = []
    identities = []
    voltage_hashes = []
    block = (base.COLS // 16) * (base.COLS // 16 + 1) // 2 + base.ROWS // 16
    for b in range(number):
        component, packet, fine, admit, kind, identity = source_recipe(
            plan, case, batch, b
        )
        raw, joint, n, k, q, mean, weight, use = base.sufficient(
            component, packet, fine, admit
        )
        voltage_hashes.append(hashlib.sha256(component.tobytes()).hexdigest())
        identities.append([kind, identity])
        for t in range(SUBS):
            co = streams["in_buf"][b].data[t, 0]
            co[block, base.COLS % 16, base.ROWS % 16, 0] = raw[t].real
            co[block, base.COLS % 16, base.ROWS % 16, 1] = -raw[t].imag
            streams["in_counts_buf"][b].data[t, 0, 0] = joint[t]
            streams["in_rficounts_buf"][b].data[t, 0] = (~fine[t]).sum()
            streams["in_plcounts_buf"][b].data[t, 0] = (~packet[t, :, 0]).sum()
            streams["in_rfiframemask_buf"][b].data[t, 0] = admit[t]
        reference.append((raw, joint, n, k, q, mean, weight, use, admit))
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
        "gps_time": {"frame0_nano": BOOT},
    }
    config["telescope"]["frame0_nano"] = BOOT
    checks = 0
    failures = []

    def check(ok, label, size=1):
        nonlocal checks
        checks += size
        if not bool(ok):
            failures.append(label)

    def close(a, b, label, weight=False):
        check(
            np.allclose(a, b, rtol=5e-6 if weight else 2e-6, atol=3e-7),
            label,
            np.size(a),
        )

    with tempfile.TemporaryDirectory(prefix="n2-composed-acc-") as tmp:
        readers = {
            key: runner.ReadChordBuffer(tmp, value) for key, value in streams.items()
        }
        for reader in readers.values():
            reader.write()
        output = PerProductDump(
            tmp, exit_after_n_files=number, num_elements=E, num_ev=0, num_freq=1
        )
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
        stage_run(stage, tmp, out / "accumulator-io.zip", out / "accumulator.log")
        frames = sorted(output.load(), key=lambda f: int(f.metadata.abs_time_idx))
        original_bytes = [
            p.read_bytes() for p in sorted(Path(tmp).glob(f"*{output.name}*.dump"))
        ]
    check(len(frames) == number, "accumulator frame count")
    for b, (frame, ref) in enumerate(zip(frames, reference)):
        raw, joint, n, k, q, mean, weight, use, admit = ref
        counts = joint[use].sum(axis=0)[FULL_R // 8, FULL_C // 8]
        check(
            np.array_equal(frame.valid_fpga_ticks, counts),
            f"accumulator counts {b}",
            len(counts),
        )
        close(frame.vis[PIND], mean, f"accumulator mean {b}")
        close(frame.weight[PIND], weight, f"accumulator weight {b}", True)
        check(
            frame.metadata.fpga_start_tick == timeline["source_start_ticks"][b],
            f"accumulator time {b}",
        )
        check(frame.metadata.frame_length_fpga_ticks == FRAME, f"accumulator span {b}")
    np.savez_compressed(
        out / "source-receipts.npz",
        raw=np.array([r[0] for r in reference]),
        counts=np.array([r[1][:, base.ROWS // 8, base.COLS // 8] for r in reference]),
        n=np.array([r[2] for r in reference]),
        k=np.array([r[3] for r in reference]),
        q=np.array([r[4] for r in reference]),
        admit=np.array([r[8] for r in reference]),
        voltage_sha256=np.array(voltage_hashes),
        generator_identities=np.array(identities),
        actual_mean=np.array([f.vis[PIND] for f in frames]),
        actual_weight=np.array([f.weight[PIND] for f in frames]),
        actual_count=np.array([f.valid_fpga_ticks[PIND] for f in frames]),
    )
    with tempfile.TemporaryDirectory(prefix="n2-composed-down-") as tmp:
        source = runner.ReadN2Buffer(tmp, frames)
        source.buffer_block[source.name]["support_mode"] = "per_product_v1"
        source.write()
        for item in source.stage_block.values():
            item["end_interrupt"] = False
        bridge = [
            p.read_bytes() for p in sorted(Path(tmp).glob("rawfileread_buf_*.dump"))
        ]
        check(
            len(bridge) == len(original_bytes)
            and all(a == b for a, b in zip(original_bytes, bridge)),
            "unchanged native accumulator replay",
            len(bridge),
        )
        save(
            out / "bridge.json",
            {
                "byte_identical": len(bridge) == len(original_bytes)
                and all(a == b for a, b in zip(original_bytes, bridge)),
                "source_sha256": [
                    hashlib.sha256(v).hexdigest() for v in original_bytes
                ],
                "replay_sha256": [hashlib.sha256(v).hexdigest() for v in bridge],
            },
        )
        output = PerProductDump(
            tmp,
            exit_after_n_files=len(timeline["complete_groups"]),
            num_elements=E,
            num_ev=0,
            num_freq=1,
        )
        downconfig = {
            "buffer_depth": 3,
            "num_elements": E,
            "num_dishes": E // 2,
            "num_polarizations": 2,
            "num_cylinders": 1,
            "num_ev": 0,
            "telescope": deepcopy(TELESCOPE),
            "gps_time": {"frame0_nano": BOOT},
            "log_level": "WARN",
        }
        stage = runner.KotekanStageTester(
            "N2TimeDownsample",
            {
                "num_bins_per_rotation": ERA_BINS,
                "max_age": 200000,
                "do_fringestop": False,
            },
            source,
            output,
            downconfig,
        )
        stage_run(stage, tmp, out / "downsampler-io.zip", out / "downsampler.log")
        down = sorted(output.load(), key=lambda f: int(f.metadata.abs_time_idx))
    check(len(down) == len(timeline["complete_groups"]), "downsampler frame count")
    rows = []
    for g, (actual, group) in enumerate(zip(down, timeline["complete_groups"])):
        indices = group["source_indices"]
        members = [frames[i] for i in indices]
        counts = np.array([f.valid_fpga_ticks for f in members], dtype=np.uint64)
        means = np.array([f.vis for f in members], dtype=np.complex128)
        weights = np.array([f.weight for f in members], dtype=np.float64)
        n = counts.sum(axis=0)
        positive = counts > 0
        unavailable = positive & ((weights <= 0) | ~np.isfinite(weights))
        expected_mean = np.divide(
            (counts * means).sum(axis=0), n, out=np.zeros(n.shape, complex), where=n > 0
        )
        terms = np.divide(
            counts.astype(float) ** 2,
            weights,
            out=np.zeros(weights.shape),
            where=positive & ~unavailable,
        )
        varsum = terms.sum(axis=0)
        expected_weight = np.divide(
            n.astype(float) ** 2,
            varsum,
            out=np.zeros(n.shape),
            where=(varsum > 0) & ~unavailable.any(axis=0),
        )
        ideal_total = sum(
            (reference[i][0][reference[i][7]].sum(axis=0) for i in indices),
            np.zeros(len(PIND), complex),
        )
        ideal_mean = np.divide(
            ideal_total, n[PIND], out=np.zeros(len(PIND), complex), where=n[PIND] > 0
        )
        check(
            np.array_equal(actual.valid_fpga_ticks, n), f"downsample counts {g}", len(n)
        )
        close(actual.vis[PIND], expected_mean[PIND], f"downsample replay mean {g}")
        close(actual.vis[PIND], ideal_mean, f"downsample ideal mean {g}")
        close(
            actual.weight[PIND],
            expected_weight[PIND],
            f"downsample precision {g}",
            True,
        )
        check(
            actual.metadata.abs_time_idx == group["relative_era_bin"],
            f"ERA identity {g}",
        )
        check(
            actual.metadata.fpga_start_tick
            == timeline["source_start_ticks"][indices[0]],
            f"group start {g}",
        )
        check(
            actual.metadata.frame_length_fpga_ticks == len(indices) * FRAME,
            f"group span {g}",
        )
        rows.append(
            {
                "actual_mean": actual.vis[PIND].copy(),
                "actual_weight": actual.weight[PIND].copy(),
                "actual_count": actual.valid_fpga_ticks[PIND].copy(),
                "ideal_mean": ideal_mean,
                "expected_mean": expected_mean[PIND],
                "expected_weight": expected_weight[PIND],
                "unavailable_upstream_count": unavailable[:, PIND].sum(axis=0),
                "zero_count_upstream_count": (~positive[:, PIND]).sum(axis=0),
                "source_frame_count": len(indices),
                "relative_era_bin": group["relative_era_bin"],
            }
        )
    np.savez_compressed(
        out / "downsample-receipts.npz",
        **{name: np.array([row[name] for row in rows]) for name in rows[0]},
    )
    save(
        out / "arithmetic.json",
        {
            "checks": checks,
            "failures": failures,
            "source_frames": number,
            "output_frames": len(down),
        },
    )
    return checks, len(failures)


def summarize(out, plan):
    moments = plan["moments"]
    rows = []
    for case in CASE_NAMES[:2]:
        files = sorted((out / "batches" / case).glob("*/downsample-receipts.npz"))
        arrays = {
            key: np.concatenate([np.load(p)[key] for p in files])
            for key in [
                "actual_mean",
                "actual_weight",
                "actual_count",
                "unavailable_upstream_count",
                "zero_count_upstream_count",
            ]
        }
        if len(arrays["actual_count"]) != plan["primary_output_groups_per_case"]:
            raise ValueError(
                f"Incomplete predetermined primary output groups for {case}"
            )
        for p in PROBES:
            count = arrays["actual_count"][:, p].astype(float)
            weights = arrays["actual_weight"][:, p].astype(float)
            target = np.divide(
                moments[p]["variance"],
                count,
                out=np.full(count.shape, np.nan),
                where=count > 0,
            )
            mse = (
                abs(
                    arrays["actual_mean"][:, p].astype(complex)
                    - complex(*moments[p]["mean"])
                )
                ** 2
            )
            good = (
                (weights > 0)
                & np.isfinite(weights)
                & (arrays["unavailable_upstream_count"][:, p] == 0)
            )
            inv = np.divide(1, weights, out=np.full(weights.shape, np.nan), where=good)
            mr = (
                float(mse.sum() / target.sum()) if np.all(np.isfinite(target)) else None
            )
            wr = (
                float(inv.sum() / target.sum())
                if np.all(np.isfinite(inv)) and np.all(np.isfinite(target))
                else None
            )
            rows.append(
                {
                    "case": case,
                    "inputs": moments[p]["inputs"],
                    "output_groups": len(count),
                    "sample_count_min": int(count.min()),
                    "sample_count_max": int(count.max()),
                    "positive_count_unavailable_precision_groups": int((~good).sum()),
                    "groups_with_zero_count_source_fragments": int(
                        np.count_nonzero(arrays["zero_count_upstream_count"][:, p])
                    ),
                    "mean_squared_error": float(mse.mean()),
                    "mean_inverse_weight": float(inv.mean()) if np.all(good) else None,
                    "mean_ideal_conditional_variance": float(target.mean()),
                    "empirical_variance_ratio": mr,
                    "reported_variance_ratio": wr,
                    "empirical_ratio_mc_se": float(
                        np.std(mse - target, ddof=1)
                        / math.sqrt(len(count))
                        / target.mean()
                    ),
                    "reported_ratio_mc_se": float(
                        np.std(inv - target, ddof=1)
                        / math.sqrt(len(count))
                        / target.mean()
                    )
                    if np.all(good)
                    else None,
                    "passes_engineering_tolerance": all(
                        x is not None and 0.85 <= x <= 1.15 for x in (mr, wr)
                    ),
                }
            )
    guard = np.load(out / "batches" / CASE_NAMES[2] / "000" / "downsample-receipts.npz")
    gp = [p for p in PROBES if base.ROWS[p] == 0]
    control = [p for p in PROBES if (base.ROWS[p], base.COLS[p]) == (8, 16)]
    guard_pass = bool(
        np.all(guard["actual_weight"][:, gp] == 0)
        & np.all(guard["actual_count"][:, gp] > 0)
        & np.all(guard["unavailable_upstream_count"][:, gp] > 0)
        & np.all(guard["actual_weight"][:, control] > 0)
        & np.all(guard["unavailable_upstream_count"][:, control] == 0)
    )
    arithmetic = [
        json.loads(p.read_text()) for p in (out / "batches").rglob("arithmetic.json")
    ]
    summary = {
        "schema": "n2-composed-precision-summary-v1",
        "plan_sha256": base.sha(out / "plan.json"),
        "primary_comparisons": 16,
        "primary_rows": rows,
        "source_frames": sum(x["source_frames"] for x in arithmetic),
        "output_frames": sum(x["output_frames"] for x in arithmetic),
        "arithmetic_checks": sum(x["checks"] for x in arithmetic),
        "arithmetic_failed_batches": sum(bool(x["failures"]) for x in arithmetic),
        "guard_output_groups": len(guard["actual_weight"]),
        "guard_unknown_product_checks": len(guard["actual_weight"]) * len(gp),
        "guard_supported_control_checks": len(guard["actual_weight"]),
        "guard_pass": guard_pass,
        "all_engineering_checks_pass": all(
            r["passes_engineering_tolerance"] for r in rows
        )
        and guard_pass
        and not any(x["failures"] for x in arithmetic),
        "scope": plan["execution"]
        + "; independent synthetic integer law and fixed engineering tolerance only; no physical or confidence coverage qualification",
    }
    save(out / "summary.json", summary)
    return summary


def validate_plan(out):
    plan = json.loads((out / "plan.json").read_text())
    if base.sha(out / "plan.json") != (out / "plan.sha256").read_text().strip():
        raise ValueError("Frozen plan hash mismatch")
    if hashes() != plan["source_sha256"]:
        raise ValueError("Frozen sources changed")
    if np.__version__ != plan["numpy_version"]:
        raise ValueError("Frozen numerical runtime changed")
    if (
        plan["primary_output_groups_per_case"]
        != plan["batches_per_primary_case"] * GROUPS_PER_BATCH
    ):
        raise ValueError("Primary output count does not match frozen batch geometry")
    if plan["timeline"] != timeline():
        raise ValueError("Frozen time/group geometry changed")
    if plan["geometry"]["source_frame_samples"] != FRAME:
        raise ValueError("Source frame is not the16384-sample upgrade geometry")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run", "smoke"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    if args.command in ("freeze", "smoke"):
        freeze(out, args.command == "smoke")
        if args.command == "freeze":
            return
    plan = validate_plan(out)
    (out / "batches").mkdir(exist_ok=False)
    started = time.monotonic()
    for ci, case in enumerate(CASE_NAMES):
        for batch in range(
            plan["batches_per_primary_case"] if ci < 2 else plan["guard_batches"]
        ):
            dest = out / "batches" / case / f"{batch:03d}"
            dest.mkdir(parents=True)
            checks, failures = run_batch(dest, plan, ci, batch)
            print(
                json.dumps(
                    {
                        "case": case,
                        "batch": batch,
                        "checks": checks,
                        "failures": failures,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    current_hashes = hashes()
    save(
        out / "source-integrity.json",
        {
            "unchanged": current_hashes == plan["source_sha256"],
            "source_sha256": current_hashes,
        },
    )
    if current_hashes != plan["source_sha256"]:
        raise ValueError(
            "Frozen sources changed during execution; raw outputs retained"
        )
    result = summarize(out, plan)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
