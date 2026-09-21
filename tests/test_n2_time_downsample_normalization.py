"""Exercise count-weighted N2 means and propagated precision in the real CPU stage."""

import copy
import ctypes
import itertools
import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from kotekan.n2buffer import N2Buffer, N2Metadata

from kotekan import runner
from kotekan import telescope as tel

pytestmark = pytest.mark.serial
D, P, EV = 4, 10, 2
TICK_NS = 2560
BOOT = 1767287750500000000
DATASET = (0x0123456789ABCDEF, 0xFEDCBA9876543210)
TELESCOPE = {
    "name": "CHIMETelescope",
    "sampling_rate_MHz": 800.0,
    "fft_length": 2048,
    "nyquist_zone": 2,
    "require_gps": False,
    "inst_lat": 50.0,
    "inst_long": -120.0,
    "frame0_nano": BOOT,
}
CASES = (
    "equal",
    "unequal",
    "zero-first",
    "zero-middle",
    "all-zero",
    "unknown-first",
    "unknown-later",
    "negative",
    "nan",
    "infinite",
    "per-product",
    "large-counts",
)


def make_frames(*, frames=110, ticks=390625, bins=17280):
    """Metadata is independently supplied from the Python telescope reference."""
    seq = np.arange(frames, dtype=np.int64) * ticks
    centers = tel.get_t_inst_ns(seq + ticks // 2, TELESCOPE)
    eops = tel.get_EOP_at_t_inst_ns(centers, TELESCOPE, False)
    rotations = tel.get_nrot_at_t_inst_ns(centers, TELESCOPE, False)
    keys = np.asarray(
        [
            int(rotations[i]) * bins + int(eops[i].ERA_deg * bins / 360)
            for i in range(frames)
        ],
        dtype=np.int64,
    )
    observed = list(dict.fromkeys(keys.tolist()))
    emitted = observed[1:-1]
    assigned = {key: CASES[i % len(CASES)] for i, key in enumerate(emitted)}
    result, records = [], {}
    for i in range(frames):
        data = np.zeros(
            ctypes.sizeof(N2Metadata) + N2Buffer.calculate_layout(D, P, EV)["size"],
            dtype=np.uint8,
        )
        frame = N2Buffer(data, skip=0, num_elements=D, num_prod=P, num_ev=EV)
        m = frame.metadata
        m.freq_id, m.freq_MHz, m.abs_time_idx = 614, 560.15625, i
        m.dataset_id[:] = DATASET
        m.fpga_start_tick, m.frame_length_fpga_ticks = int(seq[i]), ticks
        m.frame_start_time_ns = int(tel.get_t_inst_ns(seq[i], TELESCOPE))
        m.time_center_eop = eops[i]
        m.bin_eop = eops[i]
        key = int(keys[i])
        members = np.flatnonzero(keys == key)
        within = int(np.flatnonzero(members == i)[0])
        case = assigned.get(key, "equal")
        n = 16 if case == "equal" else (4, 12, 8, 20, 24)[within % 5]
        if case == "large-counts":
            n = 131072 + within * 1024
        if (
            case == "all-zero"
            or case == "zero-first"
            and within == 0
            or case == "zero-middle"
            and within == 1
        ):
            n = 0
        m.n_valid_fpga_ticks = n
        m.n_pl_fpga_ticks = ticks - n
        m.n_rfi_fpga_ticks = m.n_rfi_only_fpga_ticks = 0
        z = (
            (np.arange(P, dtype=float) + 1) / 4
            + within / 8
            + 1j * ((np.arange(P) % 3) / 8 - within / 4)
        )
        frame.vis[:] = z
        frame.weight[:] = (
            4.0 if case == "equal" else 2.0 ** ((np.arange(P) + within) % 5)
        )
        if case in ("unknown-first", "unknown-later", "negative", "nan", "infinite"):
            index = 0 if case == "unknown-first" else 1
            if within == index:
                frame.weight[:] = {
                    "negative": -2,
                    "nan": np.nan,
                    "infinite": np.inf,
                }.get(case, 0)
        if case == "per-product" and within == 1:
            frame.weight[1::2] = 0
        frame.flags[:] = 1
        frame.mask[:] = 255
        frame.gain[:] = 1
        frame.radiometer_chi2[:] = -1
        frame.eval[:] = (1, 2)
        frame.evec[:] = 1
        frame.erms[...] = -1
        if n == 0:
            # All payload is unusable, including diagnostics that can be copied from
            # the first frame. Positive-count later frames must supply fresh values.
            frame.vis[:] = complex(np.nan, np.inf)
            frame.weight[:] = np.nan
            frame.eval[:] = np.nan
            frame.evec[:] = complex(np.nan, np.inf)
            frame.erms[...] = np.nan
            frame.gain[:] = complex(np.nan, np.inf)
            frame.flags[:] = np.nan
        if key in assigned:
            records.setdefault(key, {"case": case, "indices": [], "frames": []})
            records[key]["indices"].append(i)
            records[key]["frames"].append(frame)
        result.append(frame)
    return result, records, bins


def expected(frames):
    n = sum(int(f.metadata.n_valid_fpga_ticks) for f in frames)
    mean = np.zeros(P, complex)
    weight = np.zeros(P)
    for p in range(P):
        real = imag = variance = Fraction(0)
        known = True
        for f in frames:
            count = int(f.metadata.n_valid_fpga_ticks)
            if not count:
                continue
            real += count * Fraction(float(f.vis[p].real))
            imag += count * Fraction(float(f.vis[p].imag))
            precision = float(f.weight[p])
            if precision <= 0 or not np.isfinite(precision):
                known = False
            else:
                variance += count * count / Fraction(precision)
        if n:
            mean[p] = complex(float(real / n), float(imag / n))
            if known and variance:
                value = float(Fraction(n * n) / variance)
                weight[p] = value if np.isfinite(np.float32(value)) else 0.0
    return mean, weight


def run_stage(
    tmp_path, frames, records, bins, *, failure=False, config_override=None, layout=None
):
    source = runner.ReadN2Buffer(str(tmp_path), frames)
    source.write()
    for stage in source.stage_block.values():
        stage["end_interrupt"] = False
        stage["strict_framing"] = True
    if layout:
        source.buffer_block[source.name]["n2_layout"] = layout
    output = runner.DumpN2Buffer(
        str(tmp_path), exit_after_n_files=len(records), num_elements=D, num_ev=EV
    )
    if layout:
        output.buffer_block[output.name]["n2_layout"] = layout
    config = {
        "buffer_depth": 3,
        "num_elements": D,
        "num_dishes": 2,
        "num_polarizations": 2,
        "num_ev": EV,
        "num_cylinders": 1,
        "telescope": copy.deepcopy(TELESCOPE),
        "gps_time": {"frame0_nano": BOOT},
        "log_level": "WARN",
    }
    stage_config = {
        "num_bins_per_rotation": bins,
        "max_age": 200000,
        "do_fringestop": False,
    }
    stage_config.update(config_override or {})
    test = runner.KotekanStageTester(
        "N2TimeDownsample", stage_config, source, output, config, expect_failure=failure
    )
    test.run()
    if failure:
        assert test.return_code != 0, "Malformed stream was accepted"
        return []
    return output.load()


def serialize_number(x):
    x = float(x)
    return (
        x if np.isfinite(x) else ("nan" if np.isnan(x) else "inf" if x > 0 else "-inf")
    )


def verify_and_record(actual, records, bins, label):
    ordered = list(records.items())
    assert len(actual) == len(ordered)
    destination = os.getenv("N2_DOWNSAMPLE_EVIDENCE_DIR")
    if destination:
        Path(destination, label).mkdir(parents=True, exist_ok=True)
    results = []
    for out, (key, record) in zip(actual, ordered):
        frames = record["frames"]
        mean, weight = expected(frames)
        total = sum(int(f.metadata.n_valid_fpga_ticks) for f in frames)
        start = int(frames[0].metadata.fpga_start_tick)
        span = sum(int(f.metadata.frame_length_fpga_ticks) for f in frames)
        receipt = {
            "case": record["case"],
            "absolute_rotation_bin": key,
            "bins_per_rotation": bins,
            "baseline": [0, 1],
            "input": [
                {
                    "count": int(f.metadata.n_valid_fpga_ticks),
                    "vis": [
                        serialize_number(f.vis[1].real),
                        serialize_number(f.vis[1].imag),
                    ],
                    "weight": serialize_number(f.weight[1]),
                }
                for f in frames
            ],
            "expected": {
                "vis": [mean[1].real, mean[1].imag],
                "weight": weight[1],
                "valid_ticks": total,
                "start_tick": start,
                "span_ticks": span,
            },
            "actual": {
                "vis": [
                    serialize_number(out.vis[1].real),
                    serialize_number(out.vis[1].imag),
                ],
                "weight": serialize_number(out.weight[1]),
                "valid_ticks": int(out.metadata.n_valid_fpga_ticks),
                "start_tick": int(out.metadata.fpga_start_tick),
                "span_ticks": int(out.metadata.frame_length_fpga_ticks),
                "midpoint_ns": int(out.metadata.time_center_eop.t_inst_ns),
            },
        }
        if destination:
            Path(destination, label, f"bin-{key}.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False) + "\n"
            )
        results.append((out, record, mean, weight, total, start, span))
    return results


@pytest.fixture(scope="module")
def normalized(tmp_path_factory):
    frames, records, bins = make_frames()
    actual = run_stage(
        tmp_path_factory.mktemp("downsample-normalization"), frames, records, bins
    )
    return verify_and_record(actual, records, bins, "standard")


@pytest.mark.parametrize("case", CASES)
def test_count_weighted_mean_and_precision(normalized, case):
    found = False
    for out, record, mean, weight, total, start, span in normalized:
        if record["case"] != case:
            continue
        found = True
        assert np.all(np.isfinite(out.vis))
        assert np.all(np.isfinite(out.weight))
        np.testing.assert_allclose(out.vis, mean, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(out.weight, weight, rtol=2e-6, atol=1e-7)
        assert tuple(out.metadata.dataset_id) == DATASET
        assert out.metadata.n_valid_fpga_ticks == total
        assert out.metadata.n_pl_fpga_ticks == span - total
        assert out.metadata.frame_length_fpga_ticks == span
        assert out.metadata.fpga_start_tick == start
        assert out.metadata.frame_start_time_ns == tel.get_t_inst_ns(start, TELESCOPE)
        assert (
            abs(
                out.metadata.time_center_eop.t_inst_ns
                - tel.get_t_inst_ns(start + span // 2, TELESCOPE)
            )
            <= 5
        )
        assert 0 <= out.metadata.bin_start_ERAL_deg < 360
        assert 0 <= out.metadata.bin_end_ERAL_deg < 360
        if total:
            np.testing.assert_array_equal(out.flags, 1)
            np.testing.assert_array_equal(out.gain, 1)
            np.testing.assert_allclose(out.eval, [1, 2])
            np.testing.assert_allclose(out.evec, 1)
        else:
            np.testing.assert_array_equal(out.flags, 0)
    assert found


@pytest.mark.parametrize("unsupported_flag", [0.0, np.nan])
def test_flags_fold_only_contributing_frames(tmp_path, unsupported_flag):
    frames, records, bins = make_frames()
    for record in records.values():
        supported = [f for f in record["frames"] if f.metadata.n_valid_fpga_ticks]
        for ordinal, frame in enumerate(supported):
            frame.flags[:] = 1
            frame.flags[ordinal % (D - 1)] = 0
        for frame in record["frames"]:
            if not frame.metadata.n_valid_fpga_ticks:
                frame.flags[:] = unsupported_flag
    actual = run_stage(tmp_path, frames, records, bins)
    assert len(actual) == len(records)
    for out, record in zip(actual, records.values()):
        supported = [f for f in record["frames"] if f.metadata.n_valid_fpga_ticks]
        flags = (
            np.logical_and.reduce([f.flags != 0 for f in supported])
            if supported
            else np.zeros(D, dtype=bool)
        )
        np.testing.assert_array_equal(out.flags, flags)
        mean, weight = expected(record["frames"])
        np.testing.assert_allclose(out.vis, mean, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(out.weight, weight, rtol=2e-6, atol=1e-7)


@pytest.mark.timeout(20)
def test_one_bin_per_rotation_does_not_merge_days(tmp_path):
    frames, records, bins = make_frames(frames=14, ticks=8000000000, bins=1)
    actual = run_stage(tmp_path, frames, records, bins)
    for out, _, mean, weight, total, start, span in verify_and_record(
        actual, records, bins, "rotation"
    ):
        np.testing.assert_allclose(out.vis, mean, rtol=2e-6)
        np.testing.assert_allclose(out.weight, weight, rtol=2e-6)
        assert tuple(out.metadata.dataset_id) == DATASET
        assert out.metadata.n_valid_fpga_ticks == total
        assert out.metadata.fpga_start_tick == start
        assert out.metadata.frame_length_fpga_ticks == span
    assert len(actual) >= 2
    assert all(
        b.metadata.abs_time_idx > a.metadata.abs_time_idx
        for a, b in itertools.pairwise(actual)
    )


@pytest.mark.parametrize(
    "fault",
    (
        "count",
        "count-identity",
        "time",
        "gap",
        "index",
        "frequency",
        "physical-frequency",
        "reference",
        "midpoint",
        "visibility",
        "dataset",
        "mask",
        "gain",
        "bins-zero",
        "max-age",
        "overflow",
    ),
)
def test_refuse_malformed_supported_stream(tmp_path, fault, capsys):
    frames, records, bins = make_frames(frames=25)
    # A later frame in a complete output bin exercises guards after initial alignment.
    key = next(iter(records))
    index = records[key]["indices"][1]
    frame = frames[index]
    config = {}
    if fault == "count":
        frame.metadata.n_valid_fpga_ticks = frame.metadata.frame_length_fpga_ticks + 1
    elif fault == "count-identity":
        frame.metadata.n_pl_fpga_ticks -= 1
    elif fault == "time":
        frame.metadata.frame_start_time_ns += 1
    elif fault == "gap":
        for f in frames[index:]:
            f.metadata.fpga_start_tick += 1
            f.metadata.frame_start_time_ns += TICK_NS
            seq = f.metadata.fpga_start_tick
            eop = tel.get_EOP_at_t_inst_ns(
                tel.get_t_inst_ns(
                    seq + f.metadata.frame_length_fpga_ticks // 2, TELESCOPE
                ),
                TELESCOPE,
                False,
            )
            f.metadata.time_center_eop = eop
            f.metadata.bin_eop = eop
    elif fault == "index":
        frame.metadata.abs_time_idx += 1
    elif fault == "frequency":
        frame.metadata.freq_id += 1
    elif fault == "physical-frequency":
        frame.metadata.freq_MHz += 1
    elif fault == "reference":
        frame.metadata.bin_eop.ERA_deg += 0.1
    elif fault == "midpoint":
        frame.metadata.time_center_eop.t_inst_ns += 100
    elif fault == "visibility":
        frame.vis[1] = complex(np.nan, 0)
    elif fault == "dataset":
        frame.metadata.dataset_id[0] = 1
    elif fault == "mask":
        frame.mask[0] = 0
    elif fault == "gain":
        frame.gain[0] = 2
    elif fault == "bins-zero":
        config["num_bins_per_rotation"] = 0
    elif fault == "max-age":
        config["max_age"] = -1
    elif fault == "overflow":
        frame.metadata.fpga_start_tick = (1 << 64) - 1
    run_stage(tmp_path, frames, records, bins, failure=True, config_override=config)
    diagnostic = {
        "count": "inconsistent valid/RFI/packet-loss counts",
        "count-identity": "inconsistent valid/RFI/packet-loss counts",
        "time": "FPGA/start-time identity mismatch",
        "gap": "noncontiguous or out-of-order input identity",
        "index": "noncontiguous or out-of-order input identity",
        "frequency": "frequency/layout identity changed",
        "physical-frequency": "frequency/layout identity changed",
        "reference": "EOP/time identity mismatch",
        "midpoint": "EOP/time identity mismatch",
        "visibility": "nonfinite supported visibility",
        "dataset": "dataset or RFI policy changed within an output bin",
        "mask": "input mask/gains/eigenmethod changed within a bin",
        "gain": "input mask/gains/eigenmethod changed within a bin",
        "bins-zero": "positive finite bin count, max_age and input count",
        "max-age": "positive finite bin count, max_age and input count",
        "overflow": "invalid or overflowing FPGA time interval",
    }[fault]
    assert diagnostic in capsys.readouterr().out
