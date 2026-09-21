"""CPU stage regression for mask-aware EvenOddPosDef normalization.

These deterministic correlator-sum fixtures test the production N2Accumulate
stage and its N2 output layout. Counts and first-stage masks are scalar across
inputs, as required by the current stage; this does not validate heterogeneous
baseline masks or establish a physical-noise variance calibration.
"""

import json
import os
from copy import deepcopy
from pathlib import Path

import kotekan.telescope as tel
import numpy as np
import pytest
from test_n2_accumulate import chime_tel, make_zeroed_chord_buffer

from kotekan import runner

_CASES = (
    "all-pairs-supported",
    "either-member-frame-rejected",
    "single-member-zero",
    "all-counts-zero",
    "no-complete-pair-with-positive-mean",
    "zero-pair-variance",
    "unequal-counts",
    "mixed-missing-and-rejected",
)
_NUM_ELEMENTS = 64  # Smallest count/correlation block-compatible array.
_NUM_FREQ = 3
_NUM_BINS = len(_CASES)
_SUBS_PER_BIN = 8


def _case_samples(case, scale):
    counts = np.full(_SUBS_PER_BIN, 16, dtype=np.int64)
    admit = np.ones(_SUBS_PER_BIN, dtype=np.uint8)
    if case == "either-member-frame-rejected":
        # Reject pair 1 via its odd member and pair 2 via its even member.
        admit[[3, 4]] = 0
    elif case == "single-member-zero":
        counts[:] = [16, 0, 8, 4, 0, 12, 16, 16]
    elif case == "all-counts-zero":
        counts[:] = 0
    elif case == "no-complete-pair-with-positive-mean":
        counts[:] = [16, 0, 0, 8, 4, 0, 0, 12]
    elif case == "unequal-counts":
        counts[:] = [3, 8, 5, 13, 7, 16, 2, 11]
    elif case == "mixed-missing-and-rejected":
        counts[:] = [0, 0, 4, 0, 8, 12, 16, 16]
        admit[7] = 0
    counts *= scale
    # Integer visibility means keep the supplied int32 raw sums exact. The
    # constant case explicitly exercises Q=0 despite k>0 and positive support.
    means = np.array([1, 3, 2, 1, 3, 2, 1, 2], dtype=np.float64)
    if case == "zero-pair-variance":
        means[:] = 2
    return counts, admit, means


def _reference(counts, admit, raw):
    """Independent complex128 arithmetic on pre-layout correlator sums.

    raw[t, product] is a sum, not an already-normalized visibility. Every
    admitted pair contributes to its mean; only two positive counts contribute
    one variance observation. Python integers prevent count-product overflow.
    """
    total = 0
    usable = 0
    summed = np.zeros(raw.shape[1], dtype=np.complex128)
    q = np.zeros(raw.shape[1], dtype=np.float64)
    for even in range(0, len(counts), 2):
        odd = even + 1
        if not (admit[even] and admit[odd]):
            continue
        n0, n1 = int(counts[even]), int(counts[odd])
        total += n0 + n1
        summed += raw[even] + raw[odd]
        if n0 > 0 and n1 > 0:
            usable += 1
            difference = raw[even] / n0 - raw[odd] / n1
            q += (n0 * n1 / (n0 + n1)) * np.abs(difference) ** 2
    vis = summed.conjugate() / total if total else summed * 0
    weight = np.zeros_like(q)
    if total and usable:
        np.divide(total * usable, q, out=weight, where=q > 0)
    return vis, weight, total, usable, q


def _run_accumulation(
    tmpdir_factory,
    scale=1,
    period=1,
    subintegrations_per_frame=2,
    mutation=None,
    expect_failure=False,
):
    subintegration = 16 * scale
    num_frames = _NUM_BINS * _SUBS_PER_BIN // subintegrations_per_frame
    first_seq = _SUBS_PER_BIN * subintegration * period
    frequencies = np.array([202, 614, 800], dtype=np.int32)
    telescope = deepcopy(chime_tel)
    boot_time = tel.get_unix_time_ns("2026-01-01T17:15:50.5", "utc")
    telescope["frame0_nano"] = boot_time
    config = {
        "buffer_depth": 3,
        "samples_per_data_set": subintegrations_per_frame * subintegration,
        "sub_integration_ntime": subintegration,
        "num_local_freq": _NUM_FREQ,
        "num_elements": _NUM_ELEMENTS,
        "num_polarizations": 2,
        "num_dishes": _NUM_ELEMENTS // 2,
        "num_ev": 0,
        "telescope": telescope,
        "gps_time": {"frame0_nano": boot_time},
    }

    def buffers(
        name, dtype, typename, tail_shape, tail_names, tail_scalings, extra_meta=None
    ):
        return make_zeroed_chord_buffer(
            name,
            dtype,
            typename,
            (subintegrations_per_frame, _NUM_FREQ) + tail_shape,
            ("Tc", "F") + tail_names,
            (subintegration, 1) + tail_scalings,
            first_seq,
            subintegrations_per_frame * subintegration * period,
            num_frames,
            freq_ids=frequencies,
            time_downsampling=subintegration * period,
            extra_meta=extra_meta,
        )

    # 64 inputs => 10 lower triangular 16x16 correlation blocks and one
    # 8x8 count block, with each count entry representing eight inputs.
    corr = buffers(
        "n2k_correlation",
        np.int32,
        "int32",
        (10, 16, 16, 2),
        ("DPhi", "DPlo1", "DPlo2", "C"),
        (16, 1, 1, 1),
    )
    counts = buffers(
        "n2k_counts",
        np.int32,
        "int32",
        (1, 8, 8),
        ("D8Phi", "D8Plo1", "D8Plo2"),
        (64, 8, 8),
    )
    rfi = buffers("RFImask_counts", np.int32, "int32", (), (), ())
    pl = buffers("pl_lost_counts_scalar", np.int32, "int32", (), (), ())
    mask = buffers(
        "RFIFrameMask",
        np.uint8,
        "uint8",
        (),
        (),
        (),
        {
            "rfi_frame_excision_enabled": True,
            "rfi_frame_excision_thresholds": np.array([[3.0, 0.1]], dtype=np.float32),
        },
    )

    # Construct the desired upper triangular order independently, then place
    # its conjugate-layout source in the correlator's lower triangular blocks.
    rows, cols = np.triu_indices(_NUM_ELEMENTS)
    phase = (1 + (rows + 2 * cols) % 3).astype(np.complex128)
    phase += 1j * np.where(rows == cols, 0, (2 * rows + cols) % 3 - 1)
    block_row, block_col = cols // 16, rows // 16
    block = block_row * (block_row + 1) // 2 + block_col
    block_i, block_j = cols % 16, rows % 16
    expected = {}
    for bin_idx in range(_NUM_BINS):
        for f in range(_NUM_FREQ):
            # Independent case rotation forces per-frequency counts and
            # repeated accumulator/output-ring reuse, including after k=0.
            case = _CASES[(bin_idx + 3 * f) % len(_CASES)]
            n, admitted, means = _case_samples(case, scale)
            raw = n[:, None] * means[:, None] * phase[None, :]
            reference = _reference(n, admitted, raw)
            pair_admit = admitted[::2] & admitted[1::2]
            # Exercise merged first-stage RFI and packet-loss counts. A
            # second-stage frame rejection attributes the whole pair to RFI.
            first_stage_rfi = np.zeros(_SUBS_PER_BIN, dtype=np.int64)
            if case in ("single-member-zero", "mixed-missing-and-rejected"):
                first_stage_rfi[n == 0] = subintegration
            lost = subintegration - n - first_stage_rfi
            packet_loss = int(np.sum(lost))
            rfi_per_pair = first_stage_rfi.reshape(-1, 2).sum(axis=1)
            rfi_ticks = int(
                np.sum(np.where(pair_admit, rfi_per_pair, 2 * subintegration))
            )
            expected[(bin_idx + 1, int(frequencies[f]))] = {
                "case": case,
                "reference": reference,
                "packet_loss": packet_loss * period,
                "rfi_ticks": rfi_ticks * period,
                "valid_ticks": reference[2] * period,
                "first_seq": first_seq
                + bin_idx * _SUBS_PER_BIN * subintegration * period,
                "ticks": _SUBS_PER_BIN * subintegration * period,
                "telescope": telescope,
                "subintegrations_per_frame": subintegrations_per_frame,
                "scale": scale,
                "period": period,
                "evidence": mutation is None,
            }
            for t in range(_SUBS_PER_BIN):
                frame, sub = divmod(
                    bin_idx * _SUBS_PER_BIN + t, subintegrations_per_frame
                )
                corr[frame].data[sub, f, block, block_i, block_j, 0] = raw[t].real
                corr[frame].data[sub, f, block, block_i, block_j, 1] = raw[t].imag
                counts[frame].data[sub, f] = n[t]
                pl[frame].data[sub, f] = lost[t]
                rfi[frame].data[sub, f] = first_stage_rfi[t]
                mask[frame].data[sub, f] = admitted[t]

    streams = {"corr": corr, "counts": counts, "rfi": rfi, "pl": pl, "mask": mask}
    if mutation is not None:
        mutation(streams, subintegration, period)

    work = str(tmpdir_factory.mktemp("n2-mask-normalization"))
    inputs = {}
    for name, data in (
        ("in_buf", corr),
        ("in_counts_buf", counts),
        ("in_rficounts_buf", rfi),
        ("in_plcounts_buf", pl),
        ("in_rfiframemask_buf", mask),
    ):
        inputs[name] = runner.ReadChordBuffer(work, data)
        inputs[name].write()
    output = runner.DumpN2Buffer(
        work,
        exit_after_n_files=len(expected),
        num_elements=_NUM_ELEMENTS,
        num_ev=0,
        num_freq=_NUM_FREQ,
    )
    stage = runner.KotekanStageTester(
        "N2Accumulate",
        {
            "num_freq_per_n2k_frame": _NUM_FREQ,
            "packet_loss_is_scalar": True,
            "bin_in_ERA": False,
            "num_subintegrations_per_bin": _SUBS_PER_BIN,
            "variance_mode": "EvenOddPosDef",
            "do_fringestop": False,
            "input_order": "CHIMEBeamformer",
        },
        inputs,
        output,
        config,
        expect_failure=expect_failure,
    )
    stage.run()
    if expect_failure:
        return stage
    actual = output.load()
    assert len(actual) == len(expected)
    keyed = {(int(v.metadata.abs_time_idx), int(v.metadata.freq_id)): v for v in actual}
    assert set(keyed) == set(expected)
    for key, record in expected.items():
        _write_evidence(record, key, keyed[key])
    return keyed, expected


@pytest.fixture(
    scope="module",
    params=[(1, 1, 2), (4096, 1, 2), (1, 4, 2), (1, 1, 1)],
    ids=[
        "counts-16-period-1",
        "counts-65536-period-1",
        "counts-16-period-4",
        "cross-frame-pairs",
    ],
)
def masked_accumulation(request, tmpdir_factory):
    return _run_accumulation(tmpdir_factory, *request.param)


@pytest.mark.parametrize("case", _CASES)
def test_masked_mean_counts_and_precision(masked_accumulation, case):
    actual, expected = masked_accumulation
    for key, record in expected.items():
        if record["case"] != case:
            continue
        frame = actual[key]
        vis, weight, total, usable, q = record["reference"]
        context = f"{case}, bin={key[0]}, freq={key[1]}, N={total}, k={usable}"
        assert frame.metadata.n_valid_fpga_ticks == record["valid_ticks"], context
        assert frame.metadata.n_pl_fpga_ticks == record["packet_loss"], context
        assert frame.metadata.n_rfi_fpga_ticks == record["rfi_ticks"], context
        assert frame.metadata.n_rfi_only_fpga_ticks == (
            record["ticks"] - record["valid_ticks"] - record["packet_loss"]
        ), context
        assert frame.metadata.fpga_start_tick == record["first_seq"], context
        assert frame.metadata.frame_length_fpga_ticks == record["ticks"], context
        assert frame.metadata.frame_start_time_ns == tel.get_t_inst_ns(
            record["first_seq"], record["telescope"]
        ), context
        expected_center_ns = tel.get_t_inst_ns(
            record["first_seq"] + record["ticks"] // 2, record["telescope"]
        )
        assert abs(frame.metadata.time_center_eop.t_inst_ns - expected_center_ns) <= 5
        assert np.all(np.isfinite(frame.vis)), context
        assert np.all(np.isfinite(frame.weight)), context
        assert np.all(frame.weight >= 0), context
        np.testing.assert_allclose(
            frame.vis, vis, rtol=2e-6, atol=1e-7, err_msg=context
        )
        np.testing.assert_allclose(
            frame.weight, weight, rtol=2e-5, atol=1e-7, err_msg=context
        )
        if case == "no-complete-pair-with-positive-mean":
            assert total > 0 and usable == 0
            assert np.all(np.abs(frame.vis) > 0)
            assert not np.any(frame.weight)
        if case == "zero-pair-variance":
            assert total > 0 and usable == 4 and not np.any(q)
            assert not np.any(frame.weight)


def _write_evidence(record, key, frame):
    """Optional compact receipts from actual stage output; inert in normal CI."""
    target = os.environ.get("N2_NORMALIZATION_EVIDENCE_DIR")
    if not target or not record["evidence"]:
        return
    target = Path(target)
    if record["subintegrations_per_frame"] != 2:
        target /= f"subintegrations-per-frame-{record['subintegrations_per_frame']}"
    target.mkdir(parents=True, exist_ok=True)
    vis, weight, total, usable, q = record["reference"]
    product = 2  # Upper triangular baseline (input 0, input 2), with nonzero phase.
    positive = weight > 0
    actual_vis = frame.vis[product]
    reference_vis = vis[product]
    receipt = {
        "schema": "n2-mask-normalization-cpu-v1",
        "case": record["case"],
        "bin": key[0],
        "frequency_id": key[1],
        "subintegrations_per_frame": record["subintegrations_per_frame"],
        "count_scale": record["scale"],
        "voltage_sample_period_fpga": record["period"],
        "nominal_pairs": _SUBS_PER_BIN // 2,
        "admitted_sample_count": total,
        "usable_variance_pairs": usable,
        "baseline": [0, 2],
        "Q": float(q[product]),
        "expected_visibility": [float(reference_vis.real), float(reference_vis.imag)],
        "actual_visibility": [float(actual_vis.real), float(actual_vis.imag)],
        "expected_weight": float(weight[product]),
        "actual_weight": float(frame.weight[product]),
        "nominal_pair_weight": float(total * (_SUBS_PER_BIN // 2) / q[product])
        if q[product] > 0
        else 0.0,
        "max_abs_visibility_error": float(np.max(np.abs(frame.vis - vis))),
        "max_abs_weight_error": float(np.max(np.abs(frame.weight - weight))),
        "max_rel_positive_weight_error": float(
            np.max(np.abs(frame.weight[positive] / weight[positive] - 1))
        )
        if np.any(positive)
        else 0.0,
        "all_outputs_finite": bool(
            np.all(np.isfinite(frame.vis)) and np.all(np.isfinite(frame.weight))
        ),
        "actual_fpga_metadata": {
            name: int(getattr(frame.metadata, name))
            for name in (
                "abs_time_idx",
                "fpga_start_tick",
                "frame_start_time_ns",
                "frame_length_fpga_ticks",
                "n_valid_fpga_ticks",
                "n_rfi_fpga_ticks",
                "n_rfi_only_fpga_ticks",
                "n_pl_fpga_ticks",
            )
        },
        "expected_fpga_metadata": {
            "fpga_start_tick": record["first_seq"],
            "frame_length_fpga_ticks": record["ticks"],
            "n_valid_fpga_ticks": record["valid_ticks"],
            "n_rfi_fpga_ticks": record["rfi_ticks"],
            "n_pl_fpga_ticks": record["packet_loss"],
            "n_rfi_only_fpga_ticks": record["ticks"]
            - record["valid_ticks"]
            - record["packet_loss"],
        },
    }
    name = f"counts-{record['scale']}-period-{record['period']}-bin-{key[0]}-freq-{key[1]}.json"
    (target / name).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


_REJECTIONS = (
    ("valid-plus-loss-over-period", "N2Accumulate valid count plus packet loss exceeds"),
    ("nonuniform-lower-count", "N2Accumulate requires scalar counts"),
    ("negative-count", "N2Accumulate count out of range"),
    ("overfull-count", "N2Accumulate count out of range"),
    ("count-frequency-reorder", "N2Accumulate coarse-frequency mismatch"),
    ("count-period-mismatch", "N2Accumulate time-downsampling mismatch"),
    ("skipped-frame", "N2Accumulate nonconsecutive correlation frame"),
    ("unaligned-frame", "N2Accumulate unaligned correlation frame"),
    ("nonintegral-period", "N2Accumulate invalid voltage sample period"),
    ("changed-period", "N2Accumulate voltage sample period changed"),
    ("changed-frequency-order", "N2Accumulate coarse-frequency order changed"),
    ("zero-period", "N2Accumulate invalid voltage sample period"),
    ("missing-count-frequency", "N2Accumulate missing coarse-frequency metadata"),
)


def _alter_input(streams, subintegration, period, mutation):
    count = streams["counts"][0]
    if mutation == "nonuniform-lower-count":
        count.data[0, 0, 0, 7, 0] -= 1
    elif mutation == "negative-count":
        count.data[0, 0, 0, 7, 0] = -1
    elif mutation == "overfull-count":
        count.data[0, 0, 0, 7, 0] = subintegration + 1
    elif mutation == "valid-plus-loss-over-period":
        count.data[0, 0] = subintegration
        streams["pl"][0].data[0, 0] = 1
    elif mutation == "redundant-upper-count":
        # The upper half of a diagonal count tile is not a science product.
        count.data[:, :, 0, 0, 7] = -123
    elif mutation == "count-frequency-reorder":
        count.metadata["coarse_freq"] = count.metadata["coarse_freq"][::-1].copy()
    elif mutation == "missing-count-frequency":
        del count.metadata["coarse_freq"]
    elif mutation == "count-period-mismatch":
        count.metadata["time_downsampling_fpga"] += 1
    elif mutation == "unaligned-frame":
        for data in streams.values():
            for frame in data:
                frame.metadata["fpga_seq_num"] += subintegration * period
    elif mutation == "skipped-frame":
        for data in streams.values():
            for frame in data[1:]:
                frame.metadata["fpga_seq_num"] += 2 * subintegration * period
    elif mutation == "nonintegral-period":
        for data in streams.values():
            data[0].metadata["time_downsampling_fpga"] += 1
    elif mutation == "zero-period":
        for data in streams.values():
            data[0].metadata["time_downsampling_fpga"] = 0
    elif mutation == "changed-frequency-order":
        for data in streams.values():
            data[1].metadata["coarse_freq"] = (
                data[1].metadata["coarse_freq"][::-1].copy()
            )
    elif mutation == "changed-period":
        for data in streams.values():
            data[1].metadata["time_downsampling_fpga"] *= 2
    else:
        raise AssertionError(f"Unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation, diagnostic", _REJECTIONS, ids=[v[0] for v in _REJECTIONS]
)
def test_unsupported_input_refused(tmpdir_factory, mutation, diagnostic):
    stage = _run_accumulation(
        tmpdir_factory,
        mutation=lambda streams, subintegration, period: _alter_input(
            streams, subintegration, period, mutation
        ),
        expect_failure=True,
    )
    assert stage.return_code != 0, f"Stage accepted {mutation}"
    assert diagnostic in stage.output


def test_redundant_count_triangle_is_not_science_data(tmpdir_factory):
    result = _run_accumulation(
        tmpdir_factory,
        mutation=lambda streams, subintegration, period: _alter_input(
            streams, subintegration, period, "redundant-upper-count"
        ),
    )
    # The full stage output must be unaffected, not merely start successfully.
    for case in _CASES:
        test_masked_mean_counts_and_precision(result, case)
