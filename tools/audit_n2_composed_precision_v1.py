#!/usr/bin/env python3
"""Recount composed native output independently of the experiment generator."""

import argparse
import hashlib
import inspect
import io
import itertools
import json
import math
import zipfile
from fractions import Fraction as F
from pathlib import Path

import h5py
import numpy as np
from kotekan.n2buffer import N2Buffer


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    return N2Buffer(
        bytearray(raw),
        num_elements=64,
        num_prod=2080,
        num_ev=0,
        support_mode="per_product_v1",
    )


def era_key(ns, bins):
    turns = F(779057273264000000, 10**18) + F(1002737811911354480, 10**18) * F(
        int(ns), 86400 * 10**9
    )
    return math.floor(turns * bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.result
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    stream_schema = {
        "n2k_correlation": ((8, 1, 10, 16, 16, 2), np.dtype("int32")),
        "n2k_counts": ((8, 1, 1, 8, 8), np.dtype("int32")),
        "RFImask_counts": ((8, 1), np.dtype("int32")),
        "pl_lost_counts_scalar": ((8, 1), np.dtype("int32")),
        "RFIFrameMask": ((8, 1), np.dtype("uint8")),
    }
    checks, failures, sources, outputs = 0, [], 0, 0

    def check(ok, label, size=1):
        nonlocal checks
        checks += int(size)
        if not bool(ok):
            failures.append(label)

    def close(a, b, label, rtol=2e-6, atol=3e-7):
        check(np.allclose(a, b, rtol=rtol, atol=atol), label, np.size(a))

    check(
        sha(Path(inspect.getfile(N2Buffer)).read_bytes())
        == plan["source_sha256"]["python/kotekan/n2buffer.py"],
        "actual imported native decoder identity",
    )
    check(
        sha((root / "plan.json").read_bytes())
        == (root / "plan.sha256").read_text().strip()
        == summary["plan_sha256"],
        "plan binding",
    )
    active, mus = [0, 1, 8, 16], [0j, 0j, 1 + 1j, -1j]
    pairs = list(itertools.combinations_with_replacement(range(4), 2))
    allpairs = list(itertools.combinations_with_replacement(range(64), 2))
    pidx = [allpairs.index((active[a], active[b])) for a, b in pairs]
    fr, fc = np.triu_indices(64)
    inactive = np.ones(2080, bool)
    inactive[pidx] = False
    population = [complex(a, b) for a, b in itertools.product(range(-2, 3), repeat=2)]
    moments = []
    for p, (a, b) in enumerate(pairs):
        z = np.array(
            [abs(mus[a] + x) ** 2 for x in population]
            if a == b
            else [
                (mus[a] + x) * (mus[b] + y).conjugate()
                for x, y in itertools.product(population, repeat=2)
            ]
        )
        mu, var = complex(z.mean()), float(np.mean(abs(z - z.mean()) ** 2))
        moments.append((mu, var))
        close(complex(*plan["moments"][p]["mean"]), mu, "enumerated mean", 1e-13, 1e-13)
        close(plan["moments"][p]["variance"], var, "enumerated variance", 1e-13, 1e-13)
    timeline = plan["timeline"]
    groups = timeline["complete_groups"]
    membership = {
        i: (g, j)
        for g, row in enumerate(groups)
        for j, i in enumerate(row["source_indices"])
    }
    cases = [*plan["primary_cases"], plan["guard_case"]]
    aggregate = {case: [] for case in cases}
    random_ids = set()
    for ci, case in enumerate(cases):
        nb = plan["batches_per_primary_case"] if ci < 2 else plan["guard_batches"]
        check(
            sorted(p.name for p in (root / "batches" / case).iterdir())
            == [f"{b:03d}" for b in range(nb)],
            case + ": batch coverage",
        )
        for batch in range(nb):
            folder = root / "batches" / case / f"{batch:03d}"
            label = f"{case}/{batch:03d}"
            sr, dr = (
                np.load(folder / "source-receipts.npz"),
                np.load(folder / "downsample-receipts.npz"),
            )
            with zipfile.ZipFile(folder / "accumulator-io.zip") as archive:
                raw_frames = [
                    archive.read(n)
                    for n in sorted(archive.namelist())
                    if n.endswith(".dump")
                ]
                inputs = {}
                for name in archive.namelist():
                    if name.endswith(".h5"):
                        i = int(name.rsplit(".", 2)[1])
                        with h5py.File(io.BytesIO(archive.read(name)), "r") as h:
                            d = h[next(iter(h))]
                            name_key = str(d.attrs["name"])
                            check(
                                name_key in stream_schema
                                and name_key not in inputs.setdefault(i, {}),
                                label + ": unique known stream",
                            )
                            inputs[i][name_key] = d[:]
                            if name_key in stream_schema:
                                shape, dtype = stream_schema[name_key]
                                check(
                                    d.shape == shape and d.dtype == dtype,
                                    label + ": native shape/dtype",
                                )
                            check(
                                np.array_equal(d.attrs["coarse_freq"], [614])
                                and int(d.attrs["time_downsampling_fpga"]) == 2048,
                                label + ": native frequency/sample period",
                            )
                            check(
                                int(d.attrs["fpga_seq_num"]) == (i + 1) * 16384,
                                label + ": HDF time",
                            )
            frames = [decode(raw) for raw in raw_frames]
            check(
                all(set(row) == set(stream_schema) for row in inputs.values()),
                label + ": five complete synchronized streams",
            )
            check(
                len(frames) == len(inputs) == timeline["source_frames_per_batch"],
                label + ": source population",
            )
            sources += len(frames)
            totals, keys = [], []
            for i, frame in enumerate(frames):
                lab = f"{label}/source/{i}"
                g, j = membership.get(i, (-1, -1))
                kind = (
                    0
                    if ci == 0
                    else (2 if j == 1 else 1)
                    if ci == 1
                    else (3 if j == 1 else 0)
                )
                identity = ci * 1000000 + batch * 1000 + i
                check(
                    (kind, identity) not in random_ids, lab + ": independent identity"
                )
                random_ids.add((kind, identity))
                check(
                    list(sr["generator_identities"][i]) == [kind, identity],
                    lab + ": seed recipe",
                )
                vr = np.random.Generator(
                    np.random.PCG64(
                        np.random.SeedSequence([plan["seed"], kind, identity, 0])
                    )
                )
                mr = np.random.Generator(
                    np.random.PCG64(
                        np.random.SeedSequence([plan["seed"], kind, identity, 1])
                    )
                )
                c = vr.integers(-2, 3, (8, 2048, 4, 2), dtype=np.int8)
                for a in range(4):
                    c[:, :, a, 0] += int(mus[a].real)
                    c[:, :, a, 1] += int(mus[a].imag)
                check(
                    sha(c.tobytes()) == sr["voltage_sha256"][i],
                    lab + ": all voltage bytes",
                )
                packet, fine, admit = (
                    np.ones((8, 2048, 8), bool),
                    np.ones((8, 2048), bool),
                    np.ones(8, np.uint8),
                )
                if kind == 1:
                    packet = mr.random(packet.shape) < [
                        0.92,
                        0.72,
                        0.55,
                        0.85,
                        0.80,
                        0.75,
                        0.70,
                        0.65,
                    ]
                    fine = mr.random(fine.shape) < 0.9
                    admit[int(mr.integers(8))] = 0
                elif kind == 2:
                    for t in range(8):
                        for a in range(8):
                            packet[t, :, a] = (np.arange(2048) + 3 * a + t) % (
                                5 + a % 3
                            ) != 0
                        fine[t] = (np.arange(2048) + t) % 17 != 0
                    packet[1, :, 0] = False
                    packet[3, :, 1] = False
                elif kind == 3:
                    packet[1::2, :, 0] = False
                if ci == 1 and j == 0:
                    if g % 2 == 0:
                        packet[:] = False
                    else:
                        packet[:, :, 0] = False
                stream = inputs[i]
                for key, val in (
                    ("RFIFrameMask", admit),
                    ("RFImask_counts", (~fine).sum(axis=1)),
                    ("pl_lost_counts_scalar", (~packet[:, :, 0]).sum(axis=1)),
                ):
                    check(np.array_equal(stream[key][:, 0], val), lab + ": " + key, 8)
                use = np.repeat(
                    [bool(admit[t] and admit[t + 1]) for t in range(0, 8, 2)], 2
                )
                joint = np.empty((8, 8, 8), np.int64)
                for a in range(8):
                    for b in range(8):
                        joint[:, a, b] = (packet[:, :, a] & packet[:, :, b] & fine).sum(
                            axis=1
                        )
                check(
                    np.array_equal(stream["n2k_counts"][:, 0, 0], joint),
                    lab + ": full joint counts",
                    joint.size,
                )
                counts = joint[use].sum(axis=0)[fr // 8, fc // 8]
                check(
                    np.array_equal(frame.valid_fpga_ticks, counts),
                    lab + ": full output counts",
                    2080,
                )
                raw, ns = np.zeros((8, 10), complex), np.zeros((8, 10), np.int64)
                for p, (a, b) in enumerate(pairs):
                    x = c[:, :, a, 0].astype(np.int64) + 1j * c[:, :, a, 1]
                    y = c[:, :, b, 0].astype(np.int64) + 1j * c[:, :, b, 1]
                    keep = (
                        packet[:, :, active[a] // 8]
                        & packet[:, :, active[b] // 8]
                        & fine
                    )
                    ns[:, p] = keep.sum(axis=1)
                    raw[:, p] = np.where(keep, x * y.conjugate(), 0).sum(axis=1)
                    row, col = active[a], active[b]
                    block = (col // 16) * (col // 16 + 1) // 2 + row // 16
                    co = stream["n2k_correlation"][:, 0, block, col % 16, row % 16]
                    check(
                        np.array_equal(co[:, 0] - 1j * co[:, 1], raw[:, p]),
                        lab + ": native integer products",
                        8,
                    )
                n, total = ns[use].sum(axis=0), raw[use].sum(axis=0)
                k, q = np.zeros(10, np.int64), np.zeros(10)
                for t in range(0, 8, 2):
                    for p in range(10):
                        a, b = int(ns[t, p]), int(ns[t + 1, p])
                        if use[t] and a and b:
                            delta = b * raw[t, p] - a * raw[t + 1, p]
                            q[p] += abs(delta) ** 2 / (a * b * (a + b))
                            k[p] += 1
                mean = np.divide(total, n, out=np.zeros(10, complex), where=n > 0)
                weight = np.divide(n * k, q, out=np.zeros(10), where=(q > 0) & (k > 0))
                for key, val in (
                    ("raw", raw),
                    ("counts", ns),
                    ("n", n),
                    ("k", k),
                    ("admit", admit),
                    ("actual_mean", frame.vis[pidx]),
                    ("actual_weight", frame.weight[pidx]),
                    ("actual_count", n),
                ):
                    check(
                        np.array_equal(sr[key][i], val),
                        lab + ": receipt " + key,
                        np.size(val),
                    )
                close(sr["q"][i], q, lab + ": algebraic Q", 1e-12, 1e-11)
                close(frame.vis[pidx], mean, lab + ": mean")
                close(
                    frame.weight[pidx],
                    weight,
                    lab + ": precision",
                    plan["acceptance"]["weight_rtol"],
                    plan["acceptance"]["weight_atol"],
                )
                check(
                    np.all(frame.weight[pidx][weight == 0] == 0),
                    lab + ": exact unavailable accumulator precision",
                )
                check(
                    frame.metadata.freq_id == 614
                    and frame.metadata.freq_MHz == 560.15625,
                    lab + ": accumulator frequency identity",
                )
                check(
                    np.all(frame.vis[inactive] == 0)
                    & np.all(frame.weight[inactive] == 0),
                    lab + ": inactive products",
                    2 * inactive.sum(),
                )
                check(
                    frame.metadata.fpga_start_tick == (i + 1) * 16384
                    and frame.metadata.frame_length_fpga_ticks == 16384,
                    lab + ": span",
                )
                expected_mid = (
                    plan["geometry"]["telescope"]["frame0_nano"]
                    + ((i + 1) * 16384 + 8192) * 2560
                )
                check(
                    frame.metadata.time_center_eop.t_inst_ns == expected_mid,
                    lab + ": midpoint",
                )
                keys.append(
                    era_key(
                        frame.metadata.time_center_eop.t_ut1_ns,
                        plan["geometry"]["num_bins_per_rotation"],
                    )
                )
                totals.append(total)
            unique = list(dict.fromkeys(keys))
            boot_ut1 = (
                frames[0].metadata.time_center_eop.t_ut1_ns - (16384 + 8192) * 2560
            )
            origin = era_key(boot_ut1, plan["geometry"]["num_bins_per_rotation"])
            check(
                [k - origin for k in keys] == timeline["source_bin_keys"],
                label + ": absolute ERA origin",
            )
            derived = [
                [i for i, k in enumerate(keys) if k == key] for key in unique[1:-1]
            ]
            check(
                derived == [r["source_indices"] for r in groups],
                label + ": rational native ERA grouping",
            )
            check(
                [i for i, k in enumerate(keys) if k == unique[0]]
                == timeline["dropped_initial_source_indices"],
                label + ": leading boundary",
            )
            check(
                len(keys) - 1 == timeline["final_flush_trigger_source_index"],
                label + ": flush boundary",
            )
            with zipfile.ZipFile(folder / "downsampler-io.zip") as archive:
                replay = [
                    archive.read(n)
                    for n in sorted(archive.namelist())
                    if n.startswith("rawfileread_buf_") and n.endswith(".dump")
                ]
                final = [
                    decode(archive.read(n))
                    for n in sorted(archive.namelist())
                    if "_dumpn2_" in n and n.endswith(".dump")
                ]
            bridge = json.loads((folder / "bridge.json").read_text())
            check(replay == raw_frames, label + ": exact bridge", len(replay))
            check(
                bridge["byte_identical"]
                and bridge["source_sha256"]
                == bridge["replay_sha256"]
                == [sha(r) for r in raw_frames],
                label + ": bridge hashes",
            )
            check(len(final) == len(groups), label + ": final population")
            outputs += len(final)
            for g, (frame, indices) in enumerate(zip(final, derived)):
                lab = f"{label}/final/{g}"
                members = [frames[i] for i in indices]
                counts = sum(
                    (f.valid_fpga_ticks for f in members), np.zeros(2080, np.uint64)
                )
                check(
                    np.array_equal(frame.valid_fpga_ticks, counts),
                    lab + ": all counts",
                    2080,
                )
                total = np.sum([totals[i] for i in indices], axis=0)
                n = counts[pidx]
                ideal = np.divide(total, n, out=np.zeros(10, complex), where=n > 0)
                expected_mean, expected_weight, unknown, zero = [], [], [], []
                for p in pidx:
                    re, im, var, bad, absent = F(0), F(0), F(0), 0, 0
                    for f in members:
                        ni, wi = int(f.valid_fpga_ticks[p]), float(f.weight[p])
                        if not ni:
                            absent += 1
                            continue
                        re += ni * F(float(f.vis[p].real))
                        im += ni * F(float(f.vis[p].imag))
                        if not math.isfinite(wi) or wi <= 0:
                            bad += 1
                        else:
                            var += ni * ni / F(wi)
                    count = int(counts[p])
                    expected_mean.append(
                        complex(float(re / count), float(im / count)) if count else 0j
                    )
                    expected_weight.append(
                        float(count * count / var) if var > 0 and not bad else 0.0
                    )
                    unknown.append(bad)
                    zero.append(absent)
                close(frame.vis[pidx], expected_mean, lab + ": rational bridge mean")
                close(frame.vis[pidx], ideal, lab + ": raw-sample mean")
                close(
                    frame.weight[pidx],
                    expected_weight,
                    lab + ": rational propagated precision",
                    plan["acceptance"]["weight_rtol"],
                    plan["acceptance"]["weight_atol"],
                )
                check(
                    np.all(frame.weight[pidx][np.asarray(expected_weight) == 0] == 0),
                    lab + ": exact unavailable final precision",
                )
                check(
                    frame.metadata.freq_id == 614
                    and frame.metadata.freq_MHz == 560.15625,
                    lab + ": final frequency identity",
                )
                check(
                    np.all(frame.vis[inactive] == 0)
                    & np.all(frame.weight[inactive] == 0),
                    lab + ": inactive products",
                    2 * inactive.sum(),
                )
                for key, val in (
                    ("actual_mean", frame.vis[pidx]),
                    ("actual_weight", frame.weight[pidx]),
                    ("actual_count", n),
                    ("unavailable_upstream_count", unknown),
                    ("zero_count_upstream_count", zero),
                ):
                    check(
                        np.array_equal(dr[key][g], val),
                        lab + ": receipt " + key,
                        np.size(val),
                    )
                for key, val in (
                    ("ideal_mean", ideal),
                    ("expected_mean", expected_mean),
                    ("expected_weight", expected_weight),
                ):
                    close(dr[key][g], val, lab + ": receipt " + key, 1e-12, 1e-12)
                check(
                    frame.metadata.fpga_start_tick == (indices[0] + 1) * 16384
                    and frame.metadata.frame_length_fpga_ticks == len(indices) * 16384,
                    lab + ": span",
                )
                check(
                    frame.metadata.abs_time_idx == groups[g]["relative_era_bin"],
                    lab + ": identity",
                )
                midpoint_tick = (indices[0] + 1) * 16384 + len(indices) * 8192
                check(
                    frame.metadata.time_center_eop.t_inst_ns
                    == plan["geometry"]["telescope"]["frame0_nano"]
                    + midpoint_tick * 2560,
                    lab + ": actual accumulated midpoint",
                )
                aggregate[case].append(
                    (
                        np.array(frame.vis[pidx], complex),
                        np.array(frame.weight[pidx], float),
                        np.array(n),
                        np.array(unknown),
                    )
                )
            print(
                json.dumps(
                    {"batch": label, "checks": checks, "failures": len(failures)}
                ),
                flush=True,
            )
    rows = []
    expected_rows = {
        (case, tuple(pair))
        for case in plan["primary_cases"]
        for pair in plan["primary_probes"]
    }
    declared_rows = [(r["case"], tuple(r["inputs"])) for r in summary["primary_rows"]]
    check(
        len(declared_rows) == len(set(declared_rows)) == len(expected_rows) == 8
        and set(declared_rows) == expected_rows,
        "exact primary row family",
    )
    check(
        summary["primary_comparisons"]
        == plan["acceptance"]["primary_comparisons"]
        == 16,
        "exact primary comparison count",
    )
    for row in summary["primary_rows"]:
        p = [(active[a], active[b]) for a, b in pairs].index(tuple(row["inputs"]))
        means, weights, counts, unknown = (
            np.array([r[j][p] for r in aggregate[row["case"]]]) for j in range(4)
        )
        mu, var = moments[p]
        target = var / counts
        mr = float(np.sum(abs(means - mu) ** 2) / target.sum())
        good = bool(np.all((weights > 0) & np.isfinite(weights) & (unknown == 0)))
        wr = float(np.sum(1 / weights) / target.sum()) if good else None
        close(row["empirical_variance_ratio"], mr, "ensemble error ratio", 2e-7, 1e-9)
        if wr is None:
            check(
                row["reported_variance_ratio"] is None, "unavailable primary preserved"
            )
        else:
            close(
                row["reported_variance_ratio"],
                wr,
                "ensemble reciprocal ratio",
                1e-12,
                1e-12,
            )
        passed = wr is not None and 0.85 <= mr <= 1.15 and 0.85 <= wr <= 1.15
        check(row["passes_engineering_tolerance"] == passed, "engineering acceptance")
        check(
            row["output_groups"]
            == len(means)
            == plan["primary_output_groups_per_case"],
            "fixed output population",
        )
        check(
            row["positive_count_unavailable_precision_groups"]
            == int(
                np.count_nonzero(
                    (weights <= 0) | ~np.isfinite(weights) | (unknown != 0)
                )
            ),
            "all unavailable primary groups counted",
        )
        rows.append(
            {
                "case": row["case"],
                "inputs": row["inputs"],
                "empirical_variance_ratio": mr,
                "reported_variance_ratio": wr,
                "passes": passed,
            }
        )
    gp = [
        p
        for p, (a, b) in enumerate(pairs)
        if (active[a], active[b]) in ((0, 0), (0, 1), (0, 8))
    ]
    cp = [(active[a], active[b]) for a, b in pairs].index((8, 16))
    guard_pass = True
    for _, w, n, u in aggregate[plan["guard_case"]]:
        missing_ok = bool(np.all(w[gp] == 0) & np.all(n[gp] > 0) & np.all(u[gp] > 0))
        control_ok = bool(w[cp] > 0 and u[cp] == 0)
        guard_pass &= missing_ok and control_ok
        check(missing_ok, "guard unknown", len(gp))
        check(control_ok, "guard positive control")
    ng = len(aggregate[plan["guard_case"]])
    check(
        summary["guard_pass"] == guard_pass
        and summary["guard_output_groups"] == ng
        and summary["guard_unknown_product_checks"] == ng * len(gp)
        and summary["guard_supported_control_checks"] == ng,
        "complete guard summary",
    )
    arithmetic = [
        json.loads(p.read_text()) for p in (root / "batches").rglob("arithmetic.json")
    ]
    failed_batches = sum(bool(row["failures"]) for row in arithmetic)
    check(
        summary["arithmetic_checks"] == sum(row["checks"] for row in arithmetic)
        and summary["arithmetic_failed_batches"] == failed_batches,
        "runtime arithmetic summary",
    )
    check(
        summary["all_engineering_checks_pass"]
        == (all(row["passes"] for row in rows) and guard_pass and not failed_batches),
        "all-family engineering summary",
    )
    check(
        sources == summary["source_frames"] and outputs == summary["output_frames"],
        "all native population",
    )
    for name, expected in plan["source_sha256"].items():
        check(
            sha((root / "source_snapshots" / name).read_bytes()) == expected,
            "snapshot " + name,
        )
        live_root = Path(inspect.getfile(N2Buffer)).resolve().parents[2]
        check(
            sha((live_root / name).read_bytes()) == expected,
            "unchanged evaluated runtime source " + name,
        )
    result = {
        "schema": "n2-composed-independent-audit-v1",
        "checks": checks,
        "failures": failures,
        "native_source_frames": sources,
        "native_output_frames": outputs,
        "independent_source_identities": len(random_ids),
        "primary_rows": rows,
        "scope": "All native inputs/outputs, regenerated integer voltages/masks, exact rational ERA grouping and bridge propagation, finite-law moments. Shared native decoder only; no study generator/oracle import. No physical or confidence claim.",
    }
    (args.output or root / "independent-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
