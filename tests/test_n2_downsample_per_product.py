"""Versioned per-product support survives raw IO and CPU time downsampling."""

import copy
import ctypes
import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from kotekan.n2buffer import N2Buffer, N2Metadata
from test_n2_time_downsample_normalization import (
    BOOT,
    CASES,
    DATASET,
    EV,
    TELESCOPE,
    D,
    P,
    make_frames,
)

from kotekan import runner

MODE = "per_product_v1"
pytestmark = pytest.mark.serial


def per_product_frames():
    old, records, bins = make_frames()
    result = []
    for index, source in enumerate(old):
        frame = N2Buffer.new_from_params(D, P, EV, support_mode=MODE)
        ctypes.memmove(
            ctypes.addressof(frame.metadata),
            ctypes.addressof(source.metadata),
            ctypes.sizeof(N2Metadata),
        )
        for name in (
            "n_valid_fpga_ticks",
            "n_pl_fpga_ticks",
            "n_rfi_fpga_ticks",
            "n_rfi_only_fpga_ticks",
        ):
            setattr(frame.metadata, name, 0)
        frame.valid_fpga_ticks[:] = 16
        frame.vis[:] = np.arange(P) / 4 + 1j * (np.arange(P) % 3) / 8
        frame.weight[:] = 4
        frame.flags[:] = 1
        frame.mask[:] = 255
        frame.gain[:] = 1
        frame.emethod[:] = 0
        # There is no legitimate common count for these diagnostics in this mode.
        frame.eval[:] = np.nan
        frame.evec[:] = complex(np.nan, np.inf)
        frame.erms[:] = np.nan
        frame.radiometer_chi2[:] = np.nan
        result.append(frame)
    for record in records.values():
        for within, index in enumerate(record["indices"]):
            frame = result[index]
            case = record["case"]
            counts = (np.arange(P) + 1) * (within + 1) * 4
            if case == "equal":
                counts[:] = 16
            if case == "large-counts":
                counts += 65536
            if case == "all-zero" or (case == "zero-first" and within == 0):
                counts[:] = 0
            if case == "zero-middle" and within == 1:
                counts[1::2] = 0
            if case == "per-product":
                counts[::3] = 0
            frame.valid_fpga_ticks[:] = counts
            frame.vis[:] = (
                (np.arange(P) + 1) / 4
                + within / 8
                + 1j * ((np.arange(P) % 3) / 8 - within / 4)
            )
            frame.weight[:] = old[index].weight
            frame.vis[counts == 0] = complex(np.nan, np.inf)
            frame.weight[counts == 0] = np.nan
            if not np.any(counts):
                frame.flags[:] = np.nan
                frame.gain[:] = complex(np.nan, np.inf)
        record["frames"] = [result[index] for index in record["indices"]]
    return result, records, bins


class PerProductDump(runner.DumpN2Buffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_block[self.name]["support_mode"] = MODE

    def load(self):
        return N2Buffer.load_files(
            self.output_dir + "/*" + self.name + "*.dump",
            num_elements=D,
            num_prod=P,
            num_ev=EV,
            support_mode=MODE,
        )


def run(
    tmp_path,
    frames,
    records,
    bins,
    failure=False,
    output_mode=MODE,
    input_mode=MODE,
    stage="N2TimeDownsample",
):
    source = runner.ReadN2Buffer(str(tmp_path), frames)
    source.buffer_block[source.name]["support_mode"] = input_mode
    source.write()
    for item in source.stage_block.values():
        item["end_interrupt"] = False
    output = PerProductDump(
        str(tmp_path), exit_after_n_files=len(records), num_elements=D, num_ev=EV
    )
    output.buffer_block[output.name]["support_mode"] = output_mode
    config = {
        "buffer_depth": 3,
        "num_elements": D,
        "num_dishes": 2,
        "num_polarizations": 2,
        "num_cylinders": 1,
        "num_ev": EV,
        "telescope": copy.deepcopy(TELESCOPE),
        "gps_time": {"frame0_nano": BOOT},
        "log_level": "WARN",
    }
    stage_config = {
        "num_bins_per_rotation": bins,
        "max_age": 200000,
        "do_fringestop": False,
    }
    test = runner.KotekanStageTester(
        stage, stage_config, source, output, config, expect_failure=failure
    )
    test.run()
    if failure:
        assert test.return_code != 0
        return []
    return output.load()


def oracle(frames):
    counts = [sum(int(frame.valid_fpga_ticks[p]) for frame in frames) for p in range(P)]
    mean, weight = np.zeros(P, complex), np.zeros(P)
    for p, total in enumerate(counts):
        if not total:
            continue
        real = imag = variance = Fraction(0)
        known = True
        for frame in frames:
            n = int(frame.valid_fpga_ticks[p])
            if not n:
                continue
            real += n * Fraction(float(frame.vis[p].real))
            imag += n * Fraction(float(frame.vis[p].imag))
            w = float(frame.weight[p])
            if not np.isfinite(w) or w <= 0:
                known = False
            else:
                variance += n * n / Fraction(w)
        mean[p] = complex(float(real / total), float(imag / total))
        if known and variance:
            weight[p] = float(Fraction(total * total) / variance)
    return counts, mean, weight


@pytest.fixture(scope="module")
def output(tmp_path_factory):
    frames, records, bins = per_product_frames()
    actual = run(
        tmp_path_factory.mktemp("downsample-per-product"), frames, records, bins
    )
    assert len(actual) == len(records)
    return list(zip(actual, records.values()))


@pytest.mark.parametrize("case", CASES)
def test_product_weighted_mean_and_precision(output, case):
    found = False
    for actual, record in output:
        if record["case"] != case:
            continue
        found = True
        counts, mean, weight = oracle(record["frames"])
        np.testing.assert_array_equal(actual.valid_fpga_ticks, counts)
        np.testing.assert_allclose(actual.vis, mean, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(actual.weight, weight, rtol=2e-6, atol=1e-7)
        assert (
            actual.metadata.n_valid_fpga_ticks == actual.metadata.n_pl_fpga_ticks == 0
        )
        assert (
            actual.metadata.n_rfi_fpga_ticks
            == actual.metadata.n_rfi_only_fpga_ticks
            == 0
        )
        assert tuple(actual.metadata.dataset_id) == DATASET
        assert actual.metadata.frame_length_fpga_ticks == sum(
            f.metadata.frame_length_fpga_ticks for f in record["frames"]
        )
        np.testing.assert_array_equal(actual.eval, 0)
        np.testing.assert_array_equal(actual.evec, 0)
        np.testing.assert_array_equal(actual.emethod, 0)
        np.testing.assert_array_equal(actual.erms, -1)
        np.testing.assert_array_equal(actual.radiometer_chi2, -1)
        destination = os.getenv("N2_PER_PRODUCT_EVIDENCE_DIR")
        if destination:
            path = Path(destination, "downsampler")
            path.mkdir(parents=True, exist_ok=True)
            receipt = {
                "case": case,
                "input": [
                    {
                        "counts": frame.valid_fpga_ticks.tolist(),
                        "vis": [
                            [float(z.real), float(z.imag)] if n else None
                            for z, n in zip(frame.vis, frame.valid_fpga_ticks)
                        ],
                        "weight": [
                            float(w) if np.isfinite(w) else None for w in frame.weight
                        ],
                    }
                    for frame in record["frames"]
                ],
                "counts": counts,
                "actual_counts": actual.valid_fpga_ticks.tolist(),
                "expected_vis": [[z.real, z.imag] for z in mean],
                "actual_vis": [[float(z.real), float(z.imag)] for z in actual.vis],
                "expected_weight": weight.tolist(),
                "actual_weight": actual.weight.tolist(),
            }
            (path / f"bin-{actual.metadata.abs_time_idx}.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False) + "\n"
            )
    assert found


@pytest.mark.parametrize("mode", ("scalar", MODE))
def test_python_native_round_trip_and_legacy_size(tmp_path, mode):
    frame = N2Buffer.new_from_params(D, P, EV, support_mode=mode)
    frame.metadata.dataset_id[:] = DATASET
    frame.vis[:] = np.arange(P) + 2j
    if mode == MODE:
        frame.valid_fpga_ticks[:] = np.arange(P) + (1 << 33)
    N2Buffer.to_files([frame], str(tmp_path / "input"))
    actual = N2Buffer.from_file(
        str(tmp_path / "input_0000000.dump"), D, P, EV, support_mode=mode
    )
    np.testing.assert_array_equal(actual.vis, frame.vis)
    np.testing.assert_array_equal(actual.valid_fpga_ticks, frame.valid_fpga_ticks)
    assert tuple(actual.metadata.dataset_id) == DATASET
    actual.metadata.dataset_id[0] = 17
    actual.vis[0] = 7 + 9j
    if mode == MODE:
        actual.valid_fpga_ticks[0] = (1 << 34) + 1
    N2Buffer.to_files([actual], str(tmp_path / "edited"))
    edited = N2Buffer.from_file(
        str(tmp_path / "edited_0000000.dump"), D, P, EV, support_mode=mode
    )
    assert edited.metadata.dataset_id[0] == 17
    assert edited.vis[0] == 7 + 9j
    if mode == MODE:
        assert edited.valid_fpga_ticks[0] == (1 << 34) + 1
    if mode == MODE:
        with pytest.raises(RuntimeError, match="does not match expected size"):
            N2Buffer.from_file(str(tmp_path / "input_0000000.dump"), D, P, EV)
    else:
        # Independent legacy field-size sum; no support array or new alignment.
        assert (
            N2Buffer.calculate_layout(D, P, EV)["size"]
            == 8 * P + 4 * P + 4 * D + 4 * EV + 8 * EV * D + 4 + 4 + 12 + 8 * D + D
        )


@pytest.mark.parametrize(
    "fault,diagnostic",
    (
        ("count", "per-product valid count exceeds frame interval"),
        ("global-valid", "scalar support counters must be unavailable"),
        ("global-loss", "scalar support counters must be unavailable"),
        ("visibility", "nonfinite supported visibility"),
        ("descriptor", "requires matching N2 frame descriptors"),
        ("raw-scalar", "raw file payload"),
    ),
)
def test_product_support_guards(tmp_path, capsys, fault, diagnostic):
    frames, records, bins = per_product_frames()
    frame = frames[next(iter(records.values()))["indices"][1]]
    if fault == "count":
        frame.valid_fpga_ticks[1] = frame.metadata.frame_length_fpga_ticks + 1
    if fault == "global-valid":
        frame.metadata.n_valid_fpga_ticks = 1
    if fault == "global-loss":
        frame.metadata.n_pl_fpga_ticks = 1
    if fault == "visibility":
        frame.vis[1] = complex(np.nan, 0)
    run(
        tmp_path,
        frames,
        records,
        bins,
        failure=True,
        output_mode="scalar" if fault in ("descriptor", "raw-scalar") else MODE,
        input_mode="scalar" if fault == "raw-scalar" else MODE,
    )
    assert diagnostic in capsys.readouterr().out
