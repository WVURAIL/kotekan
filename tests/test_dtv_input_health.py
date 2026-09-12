"""Compare recorded packet and mask counts with sample-level Boolean masks."""

from pathlib import Path

import numpy as np
import pytest
from kotekan.chordbuffer import ChordBuffer
from test_n2_accumulate import chime_tel

from kotekan import runner

T, F, P, D, FRAMES = 8192, 3, 2, 16, 8
FREQ = np.array([202, 614, 800], dtype=np.int32)


def inputs(period):
    streams = {key: [] for key in ("pl_buf", "rfi_buf", "dtv_buf")}
    expected = []
    for frame in range(FRAMES):
        t, f, p, g = np.indices((T, F, P, D // 8))
        present = (t // 5 + f + 2 * p + g + frame) % 7 != 0
        present[:, 0, 1, 0] = False
        if frame == 2:
            present[:] = False
        tt, ff = np.indices((T, F))
        keep = (tt + 3 * ff + frame) % 11 > 1
        keep[[63, 64, 1023, 1024], frame % F] = False
        reject = np.array([(frame + f) % 3 == 0 for f in range(F)], dtype=np.uint8)
        if frame == 0:
            reject[:] = 0
        if frame == 1:
            reject[:] = 1
        r = (~keep)[:, :, None, None]
        d = reject.astype(bool)[None, :, None, None]
        # Compute expected counts directly from the unpacked Boolean masks.
        counts = np.stack(
            [
                present.sum(axis=0),
                (~present).sum(axis=0),
                (present & r).sum(axis=0),
                (present & d).sum(axis=0),
                (present & r & d).sum(axis=0),
                (present & ~r & ~d).sum(axis=0),
                np.ones((F, P, D // 8), dtype=np.uint32),
            ],
            axis=-1,
        ).astype(np.uint32)[None]
        expected.append(counts)
        arrays = [
            (
                "pl_buf",
                "pl_mask_exp",
                "uint1x8",
                np.packbits(
                    present.reshape(128, 64, F, P, D // 8).transpose(0, 2, 3, 4, 1),
                    axis=-1,
                    bitorder="little",
                ),
                ["Thi64", "F", "P", "D8", "Tlo64"],
                [64, 1, 1, 8, 8],
                64,
            ),
            (
                "rfi_buf",
                "RFImask",
                "uint1x8",
                np.packbits(
                    keep.reshape(8, 1024, F).transpose(0, 2, 1),
                    axis=-1,
                    bitorder="little",
                ),
                ["T8hi128", "F", "T8lo128"],
                [1024, 1, 8],
                1024,
            ),
            ("dtv_buf", "dtv_mask", "int8", reject.view(np.int8), ["F"], [1], T),
        ]
        for key, name, dtype, data, dims, scales, duration in arrays:
            meta = runner.chordbuffer.get_metadata(name, dtype, dims, scales)
            meta.update(
                fpga_seq_num=(17 + frame) * T * period,
                time_downsampling_fpga=duration * period,
                coarse_freq=FREQ.copy(),
                freq_upchan_factor=np.ones(F, dtype=np.int32),
                freq_upchan_index=np.zeros(F, dtype=np.int32),
            )
            streams[key].append(ChordBuffer(data, meta))
    return streams, expected


def run(tmp_path, period, fault=None):
    streams, expected = inputs(period)
    if fault == "frequency":
        streams["pl_buf"][4].metadata["coarse_freq"] = FREQ[::-1].copy()
    elif fault == "duplicate-frequency":
        for frames in streams.values():
            frames[0].metadata["coarse_freq"] = np.array(
                [202, 202, 800], dtype=np.int32
            )
    elif fault == "missing-time":
        del streams["rfi_buf"][0].metadata["time_downsampling_fpga"]
    elif fault == "period":
        streams["pl_buf"][0].metadata["time_downsampling_fpga"] += 1
    elif fault == "noninteger-period":
        streams["dtv_buf"][0].metadata["time_downsampling_fpga"] += 1
    elif fault == "late":
        streams["rfi_buf"][4].metadata["fpga_seq_num"] -= T * period
    elif fault == "repeat":
        for frames in streams.values():
            frames[4].metadata["fpga_seq_num"] -= T * period
    elif fault == "decision":
        streams["dtv_buf"][4].data[0] = -1
    elif fault == "upchannelized":
        streams["rfi_buf"][0].metadata["freq_upchan_factor"][:] = 2
    readers = {
        key: runner.ReadChordBuffer(str(tmp_path), frames)
        for key, frames in streams.items()
    }
    for reader in readers.values():
        reader.write()
    out = runner.DumpChordBuffer(
        str(tmp_path),
        (1, F, P, D // 8, 7),
        np.uint32,
        max_frames=FRAMES,
        quantity_name="input_health_v1",
        dimnames=["Th", "F", "P", "D8", "Reason"],
        dimscalings=[T, 1, 1, 8, 1],
        input_order="CHIMEBeamformer",
    )
    stage = runner.KotekanStageTester(
        "DtvInputHealth",
        {},
        readers,
        out,
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
    if fault:
        return stage
    actual = out.load()
    assert len(actual) == FRAMES
    for i, (got, want) in enumerate(zip(actual, expected)):
        np.testing.assert_array_equal(got.data, want)
        np.testing.assert_array_equal(got.metadata["coarse_freq"], FREQ)
        np.testing.assert_array_equal(
            got.metadata["dim_names"], ["Th", "F", "P", "D8", "Reason"]
        )
        assert got.metadata["fpga_seq_num"] == (17 + i) * T * period
        assert got.metadata["time_downsampling_fpga"] == T * period
        assert got.metadata["name"] == "input_health_v1"
        # Account for every sample, counting overlapping rejections only once.
        c = got.data
        assert np.all(c[..., 0] + c[..., 1] == T)
        assert np.all(c[..., 1] + c[..., 2] + c[..., 3] - c[..., 4] + c[..., 5] == T)
        assert np.all(c[..., 6] == 1)
    return stage


@pytest.mark.parametrize("period", [1, 4])
def test_reason_counts_and_identity(tmp_path, period):
    run(tmp_path, period)


@pytest.mark.parametrize(
    "fault,diagnostic",
    [
        ("frequency", "frequency identity mismatch"),
        ("duplicate-frequency", "invalid frequency identity"),
        ("missing-time", "missing time/frequency identity"),
        ("period", "time identity mismatch"),
        ("noninteger-period", "time identity mismatch"),
        ("late", "time identity mismatch"),
        ("repeat", "discontinuous stream identity"),
        ("decision", "decision must be 0 or 1"),
        ("upchannelized", "frequency identity mismatch"),
    ],
)
def test_bad_identity_refused(tmp_path, fault, diagnostic):
    result = run(tmp_path, 4, fault)
    assert result.return_code != 0
    assert diagnostic in result.output


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_template_recorder(enabled):
    import yaml
    from jinja2 import Environment, FileSystemLoader

    template_root = Path(__file__).resolve().parents[1] / "config/fengine"
    env = Environment(loader=FileSystemLoader(str(template_root)))
    result = yaml.safe_load(
        env.get_template("chord.j2").render(
            dtv_enabled=True,
            dtv_record_input_health=enabled,
            dtv_apply_mask=False,
            dtv_input_health_dir="/tmp/health records",
        )
    )
    assert ("run_dtv_input_health" in result) == enabled
    assert "run_combine_dtv_mask" not in result
    if enabled:
        stage = result["run_dtv_input_health"]
        assert stage["rfi_buf"] == "host_rfi_RFImask_buffer"
        assert stage["pl_buf"] == "host_pl_expanded_mask_buffer"
        assert stage["out_buf"] == result["record_dtv_input_health"]["in_buf"]
        assert result["record_dtv_input_health"]["base_dir"] == "/tmp/health records"
