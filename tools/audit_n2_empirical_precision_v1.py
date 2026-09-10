#!/usr/bin/env python3
"""Reconstruct archived native N2 outputs without importing the study oracle."""

import argparse
import hashlib
import io
import itertools
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
from kotekan.n2buffer import N2Buffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    root = args.result
    plan = json.loads((root / "plan.json").read_text())
    checks, failures = 0, []

    def check(condition, label, number=1):
        nonlocal checks
        checks += number
        if not bool(condition):
            failures.append(label)

    def close(a, b, label, rtol=4e-6, atol=2e-7):
        check(np.allclose(a, b, rtol=rtol, atol=atol), label, np.size(a))

    means = [complex(*x) for x in plan["voltage_model"]["means"]]
    active = plan["geometry"]["active_inputs"]
    pairs = list(itertools.combinations_with_replacement(range(len(active)), 2))
    full_pairs = list(itertools.combinations_with_replacement(range(64), 2))
    pidx = [full_pairs.index((active[i], active[j])) for i, j in pairs]
    population = [complex(a, b) for a, b in itertools.product(range(-2, 3), repeat=2)]
    expected_moments = []
    for i, j in pairs:
        if i == j:
            z = np.array([abs(means[i] + x) ** 2 for x in population])
        else:
            z = np.array(
                [
                    (means[i] + x) * (means[j] + y).conjugate()
                    for x, y in itertools.product(population, repeat=2)
                ]
            )
        expected_moments.append(
            (complex(z.mean()), float(np.mean(abs(z - z.mean()) ** 2)))
        )
    for observed, (mu, var) in zip(plan["moments"], expected_moments):
        close(complex(*observed["mean"]), mu, "finite-law mean", atol=1e-13)
        close(observed["variance"], var, "finite-law variance", atol=1e-13)

    group_rows, group_cols = np.triu_indices(8)
    full_r, full_c = np.triu_indices(64)
    aggregate = {}
    all_native = 0
    for ci, case in enumerate([*plan["cases"], "guard"]):
        aggregate[case] = {"mean": [], "weight": [], "count": [], "k": []}
        for folder in sorted((root / "batches" / case).iterdir()):
            receipt = np.load(folder / "receipts.npz")
            with zipfile.ZipFile(folder / "native-io.zip") as archive:
                inputs = {}
                # HDF5 buffer identities, rather than order or generated names,
                # select the five actual native producer streams.
                h5_names = [n for n in archive.namelist() if n.endswith(".h5")]
                for name in h5_names:
                    index = int(name.rsplit(".", 2)[1])
                    with h5py.File(io.BytesIO(archive.read(name)), "r") as f:
                        dset = f[next(iter(f))]
                        quantity = str(dset.attrs["name"])
                        inputs.setdefault(index, {})[quantity] = dset[:]
                        check(
                            int(dset.attrs["fpga_seq_num"]) == (index + 1) * 16384,
                            f"{case}/{folder.name}/{index}: native time identity",
                        )
                native_names = sorted(
                    n for n in archive.namelist() if n.endswith(".dump")
                )
                check(
                    len(native_names) == len(receipt["realization"]),
                    "native frame count",
                )
                for b, name in enumerate(native_names):
                    all_native += 1
                    label = f"{case}/{folder.name}/{b}"
                    native = N2Buffer(
                        bytearray(archive.read(name)),
                        num_elements=64,
                        num_prod=2080,
                        num_ev=0,
                        support_mode="per_product_v1",
                    )
                    stream = inputs[b]
                    realization = int(receipt["realization"][b])
                    vrng = np.random.Generator(
                        np.random.PCG64(
                            np.random.SeedSequence([plan["seed"], ci, realization, 0])
                        )
                    )
                    mrng = np.random.Generator(
                        np.random.PCG64(
                            np.random.SeedSequence([plan["seed"], ci, realization, 1])
                        )
                    )
                    component = vrng.integers(
                        -2, 3, size=(8, 2048, 4, 2), dtype=np.int8
                    )
                    for i in range(4):
                        component[:, :, i, 0] += int(means[i].real)
                        component[:, :, i, 1] += int(means[i].imag)
                    check(
                        hashlib.sha256(component.tobytes()).hexdigest()
                        == receipt["voltage_sha256"][b],
                        label + ": voltage bytes",
                    )
                    packet = np.ones((8, 2048, 8), bool)
                    fine = np.ones((8, 2048), bool)
                    admit = np.ones(8, np.uint8)
                    if ci == 1:
                        packet = mrng.random(packet.shape) < [
                            0.92,
                            0.72,
                            0.55,
                            0.85,
                            0.80,
                            0.75,
                            0.70,
                            0.65,
                        ]
                        fine = mrng.random(fine.shape) < 0.9
                        admit[int(mrng.integers(8))] = 0
                    elif ci == 2:
                        for t in range(8):
                            for g in range(8):
                                packet[t, :, g] = (np.arange(2048) + 3 * g + t) % (
                                    5 + g % 3
                                ) != 0
                            fine[t] = (np.arange(2048) + t) % 17 != 0
                        packet[1, :, 0] = False
                        packet[3, :, 1] = False
                    elif ci == 3:
                        packet[1::2, :, 0] = False
                    check(
                        np.array_equal(stream["RFIFrameMask"][:, 0], admit),
                        label + ": frame gate",
                        8,
                    )
                    check(
                        np.array_equal(
                            stream["RFImask_counts"][:, 0], (~fine).sum(axis=1)
                        ),
                        label + ": fine counts",
                        8,
                    )
                    check(
                        np.array_equal(
                            stream["pl_lost_counts_scalar"][:, 0],
                            (~packet[:, :, 0]).sum(axis=1),
                        ),
                        label + ": scalar diagnostic",
                        8,
                    )
                    keep_pair = [
                        bool(admit[t] and admit[t + 1]) for t in range(0, 8, 2)
                    ]
                    admitted = np.repeat(keep_pair, 2)
                    joint = np.zeros((8, 8, 8), np.int64)
                    for t in range(8):
                        for i, j in zip(group_rows, group_cols):
                            joint[t, i, j] = np.count_nonzero(
                                packet[t, :, i] & packet[t, :, j] & fine[t]
                            )
                            joint[t, j, i] = joint[t, i, j]
                    check(
                        np.array_equal(stream["n2k_counts"][:, 0, 0], joint),
                        label + ": native joint counts",
                        joint.size,
                    )
                    expected_count = joint[admitted].sum(axis=0)[
                        full_r // 8, full_c // 8
                    ]
                    check(
                        np.array_equal(native.valid_fpga_ticks, expected_count),
                        label + ": actual full support",
                        2080,
                    )
                    sub_sums = np.zeros((8, len(pairs)), complex)
                    sub_counts = np.zeros((8, len(pairs)), np.int64)
                    for p, (i, j) in enumerate(pairs):
                        x = (
                            component[:, :, i, 0].astype(np.int64)
                            + 1j * component[:, :, i, 1]
                        )
                        y = (
                            component[:, :, j, 0].astype(np.int64)
                            + 1j * component[:, :, j, 1]
                        )
                        mask = (
                            packet[:, :, active[i] // 8]
                            & packet[:, :, active[j] // 8]
                            & fine
                        )
                        sub_counts[:, p] = mask.sum(axis=1)
                        sub_sums[:, p] = np.where(mask, x * y.conjugate(), 0).sum(
                            axis=1
                        )
                        row, col = active[i], active[j]
                        block = (col // 16) * (col // 16 + 1) // 2 + row // 16
                        co = stream["n2k_correlation"][:, 0, block, col % 16, row % 16]
                        check(
                            np.array_equal(co[:, 0] - 1j * co[:, 1], sub_sums[:, p]),
                            label + ": native voltage products",
                            8,
                        )
                    check(
                        np.array_equal(receipt["raw"][b], sub_sums),
                        label + ": receipt raw sums",
                        sub_sums.size,
                    )
                    check(
                        np.array_equal(receipt["counts"][b], sub_counts),
                        label + ": receipt support",
                        sub_counts.size,
                    )
                    N = sub_counts[admitted].sum(axis=0)
                    total = sub_sums[admitted].sum(axis=0)
                    expected_mean = np.divide(
                        total, N, out=np.zeros_like(total), where=N > 0
                    )
                    K = np.zeros(len(pairs), int)
                    Q = np.zeros(len(pairs))
                    for t in range(0, 8, 2):
                        for p in range(len(pairs)):
                            a, c = int(sub_counts[t, p]), int(sub_counts[t + 1, p])
                            if keep_pair[t // 2] and a and c:
                                K[p] += 1
                                # Algebraically different from differences of means.
                                delta = c * sub_sums[t, p] - a * sub_sums[t + 1, p]
                                Q[p] += abs(delta) ** 2 / (a * c * (a + c))
                    expected_weight = np.divide(
                        N * K, Q, out=np.zeros_like(Q), where=(Q > 0) & (K > 0)
                    )
                    close(native.vis[pidx], expected_mean, label + ": actual means")
                    close(
                        native.weight[pidx],
                        expected_weight,
                        label + ": actual precision",
                    )
                    for key, value in (("n", N), ("k", K)):
                        check(
                            np.array_equal(receipt[key][b], value),
                            label + ": " + key,
                            len(pairs),
                        )
                    close(
                        receipt["q"][b],
                        Q,
                        label + ": independent Q",
                        rtol=1e-12,
                        atol=1e-11,
                    )
                    check(
                        np.array_equal(receipt["actual_mean"][b], native.vis[pidx]),
                        label + ": archived means",
                        len(pairs),
                    )
                    check(
                        np.array_equal(
                            receipt["actual_weight"][b], native.weight[pidx]
                        ),
                        label + ": archived precision",
                        len(pairs),
                    )
                    check(
                        np.array_equal(
                            receipt["actual_count"][b], native.valid_fpga_ticks[pidx]
                        ),
                        label + ": archived counts",
                        len(pairs),
                    )
                    check(
                        native.metadata.fpga_start_tick == (b + 1) * 16384,
                        label + ": output time",
                    )
                    check(
                        native.metadata.frame_length_fpga_ticks == 16384,
                        label + ": output span",
                    )
                    for key, value in (
                        ("mean", native.vis[pidx]),
                        ("weight", native.weight[pidx]),
                        ("count", N),
                        ("k", K),
                    ):
                        aggregate[case][key].append(np.array(value, copy=True))
            print(
                json.dumps(
                    {
                        "audited": case + "/" + folder.name,
                        "checks": checks,
                        "failures": len(failures),
                    }
                ),
                flush=True,
            )
    summary = json.loads((root / "summary.json").read_text())
    for row in summary["primary_rows"]:
        case = row["case"]
        p = [(active[i], active[j]) for i, j in pairs].index(tuple(row["inputs"]))
        means_actual = np.array(aggregate[case]["mean"])[:, p].astype(complex)
        weights = np.array(aggregate[case]["weight"])[:, p].astype(float)
        counts = np.array(aggregate[case]["count"])[:, p]
        mu, var = expected_moments[p]
        target = var / counts
        mse = np.abs(means_actual - mu) ** 2
        ratio = mse.sum() / target.sum()
        precision_ratio = (1 / weights).sum() / target.sum()
        close(
            row["empirical_variance_ratio"],
            ratio,
            "summary empirical ratio",
            rtol=2e-7,
            atol=1e-9,
        )
        close(
            row["reported_variance_ratio"],
            precision_ratio,
            "summary reported ratio",
            rtol=1e-12,
            atol=1e-12,
        )
        check(
            row["passes_engineering_tolerance"]
            == bool(0.85 <= ratio <= 1.15 and 0.85 <= precision_ratio <= 1.15),
            "summary acceptance",
        )
    for name, digest in plan["source_sha256"].items():
        check(
            hashlib.sha256((root / "source_snapshots" / name).read_bytes()).hexdigest()
            == digest,
            "source snapshot " + name,
        )
    result = {
        "schema": "n2-empirical-independent-audit-v1",
        "checks": checks,
        "failures": failures,
        "native_frames": all_native,
        "scope": "All native serialized inputs/outputs, regenerated independent integer sample streams, Boolean support intersections, algebraically independent pair Q, finite-population moments, and primary aggregate ratios. Uses shared native N2Buffer decoding schema, not the study generator or moment/oracle functions.",
    }
    (root / "independent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
