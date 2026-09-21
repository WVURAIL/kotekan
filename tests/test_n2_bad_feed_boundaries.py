"""Bad-feed flags must describe only frames overlapping each accumulation bin."""

from copy import deepcopy

import kotekan.telescope as tel
import numpy as np
import pytest
from test_n2_accumulate import chime_tel, make_zeroed_chord_buffer

from kotekan import runner


@pytest.mark.parametrize(
    "period", [1, 4], ids=["native-ticks", "four-ticks-per-sample"]
)
@pytest.mark.parametrize(
    "subs_per_frame,subs_per_bin,first_sub,frame_masks,expected_flags",
    [
        (2, 4, 2, [0, 1, 1, 1, 1], {1: 1, 2: 1}),
        (2, 4, 0, [1, 0, 1, 1], {0: 0, 1: 1}),
        (4, 2, 0, [0, 1], {0: 0, 1: 0, 2: 1, 3: 1}),
    ],
    ids=["skipped-startup-frame", "frame-ends-bin", "frame-spans-bins"],
)
def test_bad_feed_mask_stays_in_its_bin(
    tmp_path,
    period,
    subs_per_frame,
    subs_per_bin,
    first_sub,
    frame_masks,
    expected_flags,
):
    num_elements, subintegration = 64, 16
    telescope = deepcopy(chime_tel)
    boot_time = tel.get_unix_time_ns("2026-01-01T17:15:50.5", "utc")
    telescope["frame0_nano"] = boot_time
    frame_samples = subs_per_frame * subintegration
    config = {
        "buffer_depth": 3,
        "samples_per_data_set": frame_samples,
        "sub_integration_ntime": subintegration,
        "num_local_freq": 1,
        "num_elements": num_elements,
        "num_polarizations": 2,
        "num_dishes": num_elements // 2,
        "num_ev": 0,
        "telescope": telescope,
        "gps_time": {"frame0_nano": boot_time},
    }

    def buffers(name, dtype, shape, names, scalings, extra_meta=None):
        return make_zeroed_chord_buffer(
            name,
            dtype,
            np.dtype(dtype).name,
            shape,
            names,
            scalings,
            first_sub * subintegration * period,
            frame_samples * period,
            len(frame_masks),
            freq_ids=np.array([614], dtype=np.int32),
            time_downsampling=subintegration * period,
            extra_meta=extra_meta,
        )

    corr = buffers(
        "n2k_correlation",
        np.int32,
        (subs_per_frame, 1, 10, 16, 16, 2),
        ("Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"),
        (subintegration, 1, 16, 1, 1, 1),
    )
    counts = buffers(
        "n2k_counts",
        np.int32,
        (subs_per_frame, 1, 1, 8, 8),
        ("Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"),
        (subintegration, 1, 64, 8, 8),
    )
    rfi = buffers(
        "RFImask_counts",
        np.int32,
        (subs_per_frame, 1),
        ("Tc", "F"),
        (subintegration, 1),
    )
    pl = buffers(
        "pl_lost_counts_scalar",
        np.int32,
        (subs_per_frame, 1),
        ("Tc", "F"),
        (subintegration, 1),
    )
    frame_mask = buffers(
        "RFIFrameMask",
        np.uint8,
        (subs_per_frame, 1),
        ("Tc", "F"),
        (subintegration, 1),
        {
            "rfi_frame_excision_enabled": False,
            "rfi_frame_excision_thresholds": np.array([[3.0, 0.1]], dtype=np.float32),
        },
    )
    bad_feed = buffers(
        "bad_feed_mask",
        np.int8,
        (1, 2, num_elements // 2),
        ("T", "P", "D"),
        (frame_samples, 1, 1),
    )
    for index, good in enumerate(frame_masks):
        counts[index].data.fill(subintegration if good else 0)
        pl[index].data.fill(0 if good else subintegration)
        frame_mask[index].data.fill(1)
        bad_feed[index].data.fill(good)

    inputs = {}
    for name, data in (
        ("in_buf", corr),
        ("in_counts_buf", counts),
        ("in_rficounts_buf", rfi),
        ("in_plcounts_buf", pl),
        ("in_rfiframemask_buf", frame_mask),
        ("in_bad_feed_mask_buf", bad_feed),
    ):
        inputs[name] = runner.ReadChordBuffer(str(tmp_path), data)
        inputs[name].write()
    output = runner.DumpN2Buffer(
        str(tmp_path),
        exit_after_n_files=len(expected_flags),
        num_elements=num_elements,
        num_ev=0,
        num_freq=1,
    )
    runner.KotekanStageTester(
        "N2Accumulate",
        {
            "num_freq_per_n2k_frame": 1,
            "packet_loss_is_scalar": True,
            "bin_in_ERA": False,
            "num_subintegrations_per_bin": subs_per_bin,
            "variance_mode": "EvenOddPosDef",
            "do_fringestop": False,
            "input_order": "CHIMEBeamformer",
        },
        inputs,
        output,
        config,
    ).run()
    actual = output.load()
    assert len(actual) == len(expected_flags)
    assert {int(frame.metadata.abs_time_idx) for frame in actual} == set(expected_flags)
    for frame in actual:
        bin_idx = int(frame.metadata.abs_time_idx)
        np.testing.assert_array_equal(
            frame.flags,
            np.full(num_elements, expected_flags[bin_idx]),
            err_msg=f"bin {bin_idx}",
        )
