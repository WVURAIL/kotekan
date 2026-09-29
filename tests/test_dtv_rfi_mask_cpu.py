"""Exercise mask composition and malformed metadata without requiring CUDA."""

import numpy as np
import pytest

from kotekan import runner
from test_dtv_input_health import D, F, FRAMES, P, T, inputs
from test_n2_accumulate import chime_tel


def run_mask(tmp_path, fault=None):
    source, _ = inputs(4)
    streams = {"rfi_buf": source["rfi_buf"], "dtv_buf": source["dtv_buf"]}
    if fault is not None:
        kind, value = fault
        if kind in streams:
            del streams[kind][0].metadata[value]
        elif kind == "decision":
            streams["dtv_buf"][0].data[0] = value
        elif kind == "overflow":
            for frames in streams.values():
                frames[0].metadata["fpga_seq_num"] = np.iinfo(np.int64).max - T * 4 + 1
        elif kind == "duplicate-frequency":
            for frames in streams.values():
                frames[0].metadata["coarse_freq"][:] = 202
    readers = {
        key: runner.ReadChordBuffer(str(tmp_path), frames)
        for key, frames in streams.items()
    }
    for reader in readers.values():
        reader.write()
    output = runner.DumpChordBuffer(
        str(tmp_path),
        (8, F, 128),
        np.uint8,
        max_frames=FRAMES,
        input_order="CHIMEBeamformer",
        quantity_name="RFImask",
        dimnames=["T8hi128", "F", "T8lo128"],
        dimscalings=[1024, 1, 8],
        value_type="uint1x8",
    )
    stage = runner.KotekanStageTester(
        "DtvRfiMask",
        {},
        readers,
        output,
        {
            "num_local_freq": F,
            "num_times": T,
            "num_polarizations": P,
            "num_dishes": D,
            "num_elements": P * D,
            "buffer_depth": 3,
            "telescope": chime_tel,
        },
        expect_failure=fault is not None,
    )
    stage.run()
    return stage, output, streams


def test_cpu_mask_keeps_only_samples_accepted_by_both_masks(tmp_path):
    stage, output, streams = run_mask(tmp_path)
    assert stage.return_code == 0
    actual = output.load()
    assert len(actual) == FRAMES
    for index, got in enumerate(actual):
        expected = streams["rfi_buf"][index].data.copy()
        reject = streams["dtv_buf"][index].data.astype(bool)
        expected[:, reject, :] = 0
        np.testing.assert_array_equal(got.data, expected)
        assert got.metadata["fpga_seq_num"] == (17 + index) * T * 4
        assert got.metadata["time_downsampling_fpga"] == 1024 * 4


@pytest.mark.parametrize("stream", ["rfi_buf", "dtv_buf"])
@pytest.mark.parametrize("field", ["fpga_seq_num", "time_downsampling_fpga"])
def test_missing_time_identity_has_a_clear_failure(tmp_path, stream, field):
    stage, _, _ = run_mask(tmp_path, (stream, field))
    assert stage.return_code != 0
    assert "DtvRfiMask missing time identity" in stage.output


@pytest.mark.parametrize("decision", [-1, 2, 127])
def test_nonbinary_decisions_are_rejected(tmp_path, decision):
    stage, _, _ = run_mask(tmp_path, ("decision", decision))
    assert stage.return_code != 0
    assert "DtvRfiMask decision is not 0 or 1" in stage.output


def test_sequence_overflow_is_rejected_before_publishing_output(tmp_path):
    stage, _, _ = run_mask(tmp_path, ("overflow", None))
    assert stage.return_code != 0
    assert "DtvRfiMask time alignment mismatch" in stage.output


def test_duplicate_frequency_identity_is_rejected(tmp_path):
    stage, _, _ = run_mask(tmp_path, ("duplicate-frequency", None))
    assert stage.return_code != 0
    assert "DtvRfiMask duplicate frequency identity" in stage.output
