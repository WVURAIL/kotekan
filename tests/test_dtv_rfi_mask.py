"""Test DTV mask routing and correlator counts with synthetic inputs."""

import copy
import struct

import numpy as np
import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from test_pilotproxy_integration import Pipeline, ROOT, reference, runtime

pytestmark = pytest.mark.serial
T, F, D, FRAMES = 8192, 4, 64, 8


def buffer(size):
    return dict(
        kotekan_buffer="standard",
        num_frames=4,
        frame_size=size,
        metadata_pool="main_pool",
    )


def dump(name):
    return dict(
        kotekan_stage="rawFileWrite",
        in_buf=f"host_{name}_buffer",
        file_name=name,
        file_ext="raw",
        prefix_hostname=False,
        num_frames_per_file=1,
        allow_ndarray=True,
        exit_after_n_files=FRAMES,
    )


def send(name, size):
    return {
        "gpu_0": dict(
            kotekan_stage="cudaProcess",
            gpu_id=0,
            in_buffers={f"host_{name}_buffer": f"host_{name}_buffer"},
            out_buffers={f"host_{name}_ringbuffer": f"host_{name}_ringbuffer"},
            commands=[
                dict(
                    name="cudaCopyToRingbuffer",
                    input_size=size,
                    ring_buffer_size=4 * size,
                    in_buf=f"host_{name}_buffer",
                    signal_buf=f"host_{name}_ringbuffer",
                    gpu_mem_output=f"{name}_buffer",
                )
            ],
        )
    }


def raw_payload(path, shape):
    with path.open("rb") as stream:
        offset = struct.unpack("<I", stream.read(4))[0] + 4
    return np.memmap(path, mode="r+", dtype=np.uint8, offset=offset, shape=shape)


def setup_pipeline(
    tmp_path,
    runtime,
    *,
    real_detector=False,
    fault=None,
    fine_support=False,
    num_times=T,
    reject_finite=False,
    plcounts=False,
):
    T = num_times
    pipeline = Pipeline(tmp_path, runtime)
    pipeline.calibrate()
    freq_ids = pipeline.generator["manual_freq_ids"]
    if reject_finite:
        # Keep this synthetic threshold separate from the archive-review bundle.
        for profile in pipeline.bundle["profiles"]:
            if (
                profile.get("receiver_channel_id", profile.get("chord_channel_id"))
                == freq_ids[2]
            ):
                profile["fine_calibration"]["cfar_multiplier_q16"] = 1 << 14
    cfg = pipeline.config
    cfg.update(num_dishes=D, num_frequencies=F, num_times=T)
    pipeline.generator["array_shape"] = [T, F, 2, D]
    # Generate metadata-bearing input files first, then introduce reproducible
    # packet loss and masks and replay through the actual production stages.
    generated = [
        (
            "rfi_RFImask",
            T * F // 8,
            [T // 1024, F, 128],
            ["T8hi128", "F", "T8lo128"],
            [1024, 1, 8],
            "const1x8",
            "RFImask",
            4096,
        ),
        (
            "pl_expanded_mask",
            T * F * 2 * D // 64,
            [T // 64, F, 2, 8, 8],
            ["Thi64", "F", "P", "D8", "Tlo64"],
            [64, 1, 1, 8, 8],
            "const1x8",
            "pl_mask_exp",
            256,
        ),
        ("dtv_mask", F, [F], ["F"], [1], "const8", "dtv_mask", T * 4),
    ]
    if plcounts:
        generated.append(
            (
                "n2k_plcounts",
                F * 4,
                [1, F],
                ["Tc", "F"],
                [T, 1],
                "const32",
                "pl_lost_counts_scalar",
                T * 4,
            )
        )
    for name, size, shape, dims, scales, kind, quantity, period in generated:
        cfg[f"host_{name}_buffer"] = buffer(size)
        cfg[f"gen_{name}"] = dict(
            kotekan_stage="testDataGen",
            out_buf=f"host_{name}_buffer",
            type=kind,
            value=0,
            name=quantity,
            array_shape=shape,
            dim_name=dims,
            dim_scaling=scales,
            first_frame_index=17,
            samples_per_data_set=T * 4,
            num_frames=FRAMES,
            wait=False,
            manual_freq_ids=freq_ids,
            meta_time_downsample_factor=period,
        )
        cfg[f"dump_{name}"] = dump(name)
    if fine_support:
        cfg["host_dtv_fine_support_buffer"] = buffer(F * 2 * 4)
        cfg["gen_dtv_fine_support"] = dict(cfg["gen_dtv_mask"])
        cfg["gen_dtv_fine_support"].update(
            out_buf="host_dtv_fine_support_buffer",
            type="const32",
            name="dtv_fine_support",
            array_shape=[F, 2],
            dim_name=["F", "S"],
            dim_scaling=[1, 1],
        )
        cfg["dump_dtv_fine_support"] = dump("dtv_fine_support")
    # Frequency mismatch must fail even when the byte layout is identical.
    if fault == "frequency":
        cfg["gen_dtv_mask"]["manual_freq_ids"] = [
            freq_ids[1],
            freq_ids[0],
            freq_ids[2],
            freq_ids[3],
        ]
    full = copy.deepcopy(cfg)
    for key in (
        "run_send_voltage",
        "run_dtv_detector",
        "dump_dtv_powers",
        "host_voltage_ringbuffer",
        "host_dtv_powers_buffer",
        "frame_arrival_period",
        "pilot_profiles_path",
        "weights_path",
        "num_frequencies",
    ):
        cfg.pop(key, None)
    code, log = pipeline.run()
    assert code == 0, log[-10000:]
    (tmp_path / "input").mkdir()
    for path in (tmp_path / "out").glob("*.raw"):
        path.rename(tmp_path / "input" / path.name)
    keep_masks, pl_masks, decisions = [], [], []
    for index in range(FRAMES):

        def source(name):
            return sorted((tmp_path / "input").glob(f"{name}_*.raw"))[index]

        # Nonuniform in both time and frequency, with accepted and rejected
        # samples on both sides of every 1024-sample layout boundary.
        t, f = np.indices((T, F))
        keep = (t + 3 * f + index) % 11 > 1
        keep[1023:1025, index % F] = False
        raw_payload(source("rfi_RFImask"), (T // 1024, F, 128))[:] = np.packbits(
            keep.reshape(T // 1024, 1024, F).transpose(0, 2, 1),
            axis=-1,
            bitorder="little",
        )
        t, f, group = np.indices((T, F, 16))
        pl = (t // 32 + 5 * f + group + index) % 7 != 0
        # Include wholly absent input groups and a wholly absent time block.
        pl[:, 0, 3] = False
        pl[4096:4160] = False
        raw_payload(source("pl_expanded_mask"), (T // 64, F, 2, 8, 8))[:] = np.packbits(
            pl.reshape(T // 64, 64, F, 2, 8).transpose(0, 2, 3, 4, 1),
            axis=-1,
            bitorder="little",
        )
        voltage = raw_payload(source("voltage"), (T, F, 128))
        voltage[~np.repeat(pl, 8, axis=2)] = 0x88  # offset-binary complex zero
        if fine_support and real_detector and index == 4:
            # No reference support on a finite-threshold channel.
            voltage[:, 2, :] = 0x88
        decision = np.array([(index + f) % 3 == 0 for f in range(F)], dtype=np.uint8)
        if index == 0:
            decision[:] = 0
        if index == 1:
            decision[:] = 1
        raw_payload(source("dtv_mask"), (F,))[:] = decision
        if plcounts:
            lost = np.array([10 * index + f for f in range(F)], dtype="<i4")
            raw_payload(source("n2k_plcounts"), (F * 4,))[:] = lost.view(np.uint8)
        if fine_support:
            support = np.array([[1, 124], [0, 0], [-1, -1], [1, 126]], dtype="<i4")
            if index % 2:
                support[[0, 1]] = support[[1, 0]]
            if fault == "support_value":
                support[0] = [2, 124]
            if fault == "support_count":
                support[0] = [1, 0]
            raw_payload(source("dtv_fine_support"), (F * 2 * 4,))[:] = support.view(
                np.uint8
            ).ravel()
            if index == 4 and fault in ("support_time", "support_period"):
                path = source("dtv_fine_support")
                data = bytearray(path.read_bytes())
                if fault == "support_time":
                    seq = struct.unpack_from("<q", data, 36)[0]
                    struct.pack_into("<q", data, 36, seq + T * 4)
                else:
                    struct.pack_into("<i", data, 44, T * 8)
                path.write_bytes(data)
        if index == 4 and fault in ("late", "missing", "period"):
            path = source("dtv_mask")
            data = bytearray(path.read_bytes())
            if fault in ("late", "missing"):
                seq = struct.unpack_from("<q", data, 36)[0]
                struct.pack_into(
                    "<q", data, 36, seq + (-1 if fault == "late" else T * 4)
                )
            else:
                struct.pack_into("<i", data, 44, T * 8)
            path.write_bytes(data)
        keep_masks.append(keep)
        pl_masks.append(pl)
        decisions.append(decision)
    pipeline.config = cfg = full
    input_names = ["voltage", "rfi_RFImask", "pl_expanded_mask", "dtv_mask"]
    if fine_support:
        input_names.append("dtv_fine_support")
    if plcounts:
        input_names.append("n2k_plcounts")
    for name in input_names:
        cfg.pop(f"gen_{name}", None)
        cfg.pop(f"dump_{name}", None)
        cfg[f"read_{name}"] = dict(
            kotekan_stage="rawFileRead",
            buf=f"host_{name}_buffer",
            base_dir="input",
            file_name=name,
            file_ext="raw",
            prefix_hostname=False,
            end_interrupt=False,
        )
    cfg.pop("samples_per_data_set")
    cfg.pop("num_gen_frames")
    for key in ("dump_dtv_mask", "dump_dtv_powers"):
        if key in cfg:
            cfg[key]["exit_after_n_files"] = FRAMES
    if real_detector:
        cfg.pop("read_dtv_mask")
        cfg["dump_dtv_mask"] = dump("dtv_mask")
        if fine_support:
            cfg.pop("read_dtv_fine_support")
            cfg["dump_dtv_fine_support"] = dump("dtv_fine_support")
            gpu = cfg["run_dtv_detector"]["gpu_0"]
            gpu["out_buffers"]["host_dtv_fine_support"] = "host_dtv_fine_support_buffer"
            gpu["commands"][0].update(
                dtv_fine_support_name="dtv_fine_support",
                permanent_mask_freq_ids=[freq_ids[0]],
            )
            gpu["commands"].append(
                dict(
                    name="cudaOutputData",
                    gpu_mem="dtv_fine_support_buffer",
                    out_buf="host_dtv_fine_support",
                )
            )
            # The reject-all limit needs no invented finite calibration.
            bound = [
                row
                for row in pipeline.bundle["profiles"]
                if row.get("receiver_channel_id", row.get("chord_channel_id"))
                == freq_ids[0]
            ]
            assert len(bound) == 1
            bound[0]["fine_calibration"]["status"] = "pending_campaign"
    else:
        cfg.pop("run_dtv_detector")
        cfg.pop("dump_dtv_powers")
        cfg.pop("host_dtv_powers_buffer")
        for key in (
            "pilot_profiles_path",
            "weights_path",
        ):
            cfg.pop(key)
    cfg["host_dtv_RFImask_buffer"] = buffer(T * F // 8)
    cfg["combine"] = dict(
        kotekan_stage="DtvRfiMask",
        num_local_freq=F,
        rfi_buf="host_rfi_RFImask_buffer",
        dtv_buf="host_dtv_mask_buffer",
        out_buf="host_dtv_RFImask_buffer",
    )
    if fine_support:
        cfg["host_dtv_fine_support_buffer"] = dict(
            kotekan_buffer="ndarray",
            num_frames=4,
            value_type="int32",
            extents=[F, 2],
            quantity_name="dtv_fine_support",
            dimnames=["F", "S"],
            dimscalings=[1, 1],
            metadata_pool="main_pool",
        )
        cfg["combine"].update(
            fine_support_buf="host_dtv_fine_support_buffer",
            permanent_mask_freq_ids=[freq_ids[0]] if real_detector else [freq_ids[3]],
            require_fine_freq_ids=(
                [freq_ids[2]] if real_detector else [freq_ids[0], freq_ids[1]]
            ),
        )
    for name, dtype, extents, quantity, dims, scales in (
        (
            "rfi_RFImask",
            "uint1x8",
            [T // 1024, F, 128],
            "RFImask",
            ["T8hi128", "F", "T8lo128"],
            [1024, 1, 8],
        ),
        (
            "dtv_RFImask",
            "uint1x8",
            [T // 1024, F, 128],
            "RFImask",
            ["T8hi128", "F", "T8lo128"],
            [1024, 1, 8],
        ),
        ("dtv_mask", "int8", [F], "dtv_mask", ["F"], [1]),
    ):
        cfg[f"host_{name}_buffer"] = dict(
            kotekan_buffer="ndarray",
            num_frames=4,
            value_type=dtype,
            extents=extents,
            quantity_name=quantity,
            dimnames=dims,
            dimscalings=scales,
            metadata_pool="main_pool",
        )
    cfg["dump_combined"] = dump("dtv_RFImask")
    cfg["host_rficounts_buffer"] = dict(
        kotekan_buffer="ndarray",
        num_frames=4,
        value_type="int32",
        extents=[1, F],
        quantity_name="RFImask_counts",
        dimnames=["Tc", "F"],
        dimscalings=[T, 1],
        metadata_pool="main_pool",
    )
    cfg["count_rfi"] = dict(
        kotekan_stage="RfiMaskSum",
        num_local_freq=F,
        samples_per_data_set=T,
        sub_integration_ntime=T,
        rfi_downsampling_factor=1,
        in_buf="host_dtv_RFImask_buffer",
        out_buf="host_rficounts_buffer",
    )
    cfg["dump_rficounts"] = dump("rficounts")
    for name, size in (
        ("dtv_RFImask", T * F // 8),
        ("pl_expanded_mask", T * F * 2 * D // 64),
    ):
        cfg[f"host_{name}_ringbuffer"] = dict(
            kotekan_buffer="ring", ring_buffer_size=4 * size, metadata_pool="main_pool"
        )
        cfg[f"send_{name}"] = send(name, size)
    for product, command, input_name, size in (
        ("n2k_correlation", "cudaCorrelator", "voltage", F * 36 * 16 * 16 * 2 * 4),
        ("n2k_counts", "cudaPL1bitCorrelator", "pl_expanded_mask", F * 3 * 8 * 8 * 4),
    ):
        cfg[f"host_{product}_buffer"] = buffer(size)
        cmd = dict(
            name=command,
            num_elements=128,
            num_local_freq=F,
            sub_integration_ntime=T,
            rfi_RFImask_name="dtv_RFImask",
            **{f"{input_name}_name": input_name, f"{product}_name": product},
        )
        if command == "cudaPL1bitCorrelator":
            cmd.pop("num_elements")
            cmd.pop("num_local_freq")
        cfg[f"run_{product}"] = {
            "gpu_0": dict(
                kotekan_stage="cudaProcess",
                gpu_id=0,
                in_buffers={
                    f"host_{input_name}_ringbuffer": f"host_{input_name}_ringbuffer",
                    "host_dtv_RFImask_ringbuffer": "host_dtv_RFImask_ringbuffer",
                },
                out_buffers={f"host_{product}_buffer": f"host_{product}_buffer"},
                commands=[
                    cmd,
                    dict(name="cudaSyncOutput"),
                    dict(
                        name="cudaOutputData",
                        gpu_mem=f"{product}_buffer",
                        out_buf=f"host_{product}_buffer",
                    ),
                ],
            )
        }
        cfg[f"dump_{product}"] = dump(product)
    if plcounts:
        # Missing decisions count as lost samples, and the RFI count leaves them out.
        for name, dtype, extents, quantity, dims, scales in (
            (
                "n2k_plcounts",
                "int32",
                [1, F],
                "pl_lost_counts_scalar",
                ["Tc", "F"],
                [T, 1],
            ),
            (
                "dtv_plcounts",
                "int32",
                [1, F],
                "pl_lost_counts_scalar",
                ["Tc", "F"],
                [T, 1],
            ),
            (
                "dtv_rfi_count",
                "uint1x8",
                [T // 1024, F, 128],
                "RFImask",
                ["T8hi128", "F", "T8lo128"],
                [1024, 1, 8],
            ),
        ):
            cfg[f"host_{name}_buffer"] = dict(
                kotekan_buffer="ndarray",
                num_frames=4,
                value_type=dtype,
                extents=extents,
                quantity_name=quantity,
                dimnames=dims,
                dimscalings=scales,
                metadata_pool="main_pool",
            )
        cfg["combine"].update(
            plcounts_in_buf="host_n2k_plcounts_buffer",
            plcounts_out_buf="host_dtv_plcounts_buffer",
            rfi_count_buf="host_dtv_rfi_count_buffer",
        )
        cfg["count_rfi"]["in_buf"] = "host_dtv_rfi_count_buffer"
        cfg["dump_dtv_plcounts"] = dump("dtv_plcounts")
    return pipeline, keep_masks, pl_masks, decisions


def triangles(matrix, block):
    n = matrix.shape[-1] // block
    return np.stack(
        [
            matrix[..., i * block : (i + 1) * block, j * block : (j + 1) * block]
            for i in range(n)
            for j in range(i + 1)
        ],
        axis=-3,
    )


@pytest.mark.parametrize(
    "real_detector,fine_support",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_mask_visibility_and_counts_across_ring_wrap(
    tmp_path,
    runtime,
    real_detector,
    fine_support,
    num_times=T,
    reject_finite=False,
    control=False,
    missing=False,
):
    T = num_times
    pipeline, masks, packet_masks, decisions = setup_pipeline(
        tmp_path,
        runtime,
        real_detector=real_detector,
        fine_support=fine_support,
        num_times=T,
        reject_finite=reject_finite,
    )
    freq_ids = pipeline.generator["manual_freq_ids"]
    combine = pipeline.config["combine"]
    if control:
        combine["control_freq_ids"] = [freq_ids[2]]
    if missing:
        # Frequency 2's support rows are (-1, -1): missing decisions.
        combine["require_fine_freq_ids"] = combine["require_fine_freq_ids"] + [
            freq_ids[2]
        ]
        combine["missing_decision_action"] = "mask"
    code, log = pipeline.run()
    assert code == 0, log[-12000:]

    def read(folder, name, size):
        result = reference.read_raw_frames(
            str(tmp_path / folder / f"{name}_*.raw"), size
        )
        assert len(result) == FRAMES
        return result

    voltage = read("input", "voltage", T * F * 128)
    combined = read("out", "dtv_RFImask", T * F // 8)
    vis = read("out", "n2k_correlation", F * 36 * 16 * 16 * 2 * 4)
    counts = read("out", "n2k_counts", F * 3 * 8 * 8 * 4)
    rficounts = read("out", "rficounts", F * 4)
    if real_detector:
        decisions = [r.payload for r in read("out", "dtv_mask", F)]
        # Support and permanent masking exclude samples independently of a raw
        # detection. The low finite word also exercises supported RFI rejection.
        expected_decisions = {0} if fine_support and not reject_finite else {0, 1}
        assert set(np.concatenate(decisions)) == expected_decisions
    if real_detector and fine_support:
        support_frames = read("out", "dtv_fine_support", F * 2 * 4)
        support_rows = [
            frame.payload.view("<i4").reshape(F, 2) for frame in support_frames
        ]
        assert all(tuple(row[0]) == (-1, -1) for row in support_rows)
        assert tuple(support_rows[4][2]) == (0, 0)
        assert decisions[4][2] == 0  # Invalid remains distinct from an RFI detection.
        assert any(row[2, 0] == 1 for row in support_rows)
        if reject_finite:
            assert any(
                row[2, 0] == 1 and decision[2] == 1
                for row, decision in zip(support_rows, decisions)
            )
        assert all(
            frame.fpga_seq_num == (17 + i) * T * 4
            and frame.time_downsampling_fpga == T * 4
            for i, frame in enumerate(support_frames)
        )
    for index in range(FRAMES):
        good = masks[index] & ~decisions[index].astype(bool)[None, :]
        if fine_support:
            if real_detector:
                good[:, support_rows[index][:, 0] == 0] = False
                good[:, 0] = False
            else:
                # Alternate unsupported rank across ring wraps. The last receiver
                # frequency is at the permanent-mask limit even with raw mask zero.
                good[:, 0 if index % 2 else 1] = False
                good[:, 3] = False
        if control:
            good[:, 2] = masks[index][:, 2]  # the control's decision is never applied
        if missing:
            good[:, 2] = False
        packed = np.packbits(
            good.reshape(T // 1024, 1024, F).transpose(0, 2, 1),
            axis=-1,
            bitorder="little",
        )
        np.testing.assert_array_equal(
            combined[index].payload.reshape(T // 1024, F, 128), packed
        )
        np.testing.assert_array_equal(
            rficounts[index].payload.view("<i4"), (~good).sum(axis=0)
        )
        for product, period in ((combined, 4096), (vis, T * 4), (counts, T * 4)):
            assert product[index].fpga_seq_num == (17 + index) * T * 4
            assert product[index].time_downsampling_fpga == period
        raw = voltage[index].payload.reshape(T, F, 128).astype(np.int16)
        z = ((raw >> 4) - 8 + 1j * ((raw & 15) - 8)).astype(np.complex64)
        z *= good[:, :, None]
        expected_vis = np.stack([z[:, f].T @ z[:, f].conj() for f in range(F)])
        got_vis = vis[index].payload.view("<i4").reshape(F, 36, 16, 16, 2)
        want_vis = triangles(expected_vis, 16)
        np.testing.assert_array_equal(got_vis[..., 0], want_vis.real)
        np.testing.assert_array_equal(got_vis[..., 1], want_vis.imag)
        p = (packet_masks[index] & good[:, :, None]).astype(np.float64)
        expected_counts = np.stack([p[:, f].T @ p[:, f] for f in range(F)])
        got_counts = counts[index].payload.view("<i4").reshape(F, 3, 8, 8)
        np.testing.assert_array_equal(got_counts, triangles(expected_counts, 8))
        # Products without accepted samples have no measured visibility.
        assert np.any(expected_counts == 0)
        if not real_detector and not control and index == 1:
            assert not got_counts.any() and not got_vis.any()
    if control:
        assert any(decision[2] for decision in decisions)
    if missing:
        assert "required fine threshold not evaluated" in log, log[-12000:]


@pytest.mark.parametrize("fault", ["frequency", "late", "missing", "period"])
def test_refuse_wrong_or_missing_decision_identity(tmp_path, runtime, fault):
    pipeline, *_ = setup_pipeline(tmp_path, runtime, fault=fault)
    code, log = pipeline.run()
    assert code != 0, log[-12000:]
    assert "DtvRfiMask" in log and ("mismatch" in log), log[-12000:]
    assert len(list((tmp_path / "out").glob("dtv_RFImask_*.raw"))) < FRAMES


@pytest.mark.parametrize(
    "fault", ["support_time", "support_period", "support_value", "support_count"]
)
def test_refuse_invalid_support_product(tmp_path, runtime, fault):
    pipeline, *_ = setup_pipeline(tmp_path, runtime, fault=fault, fine_support=True)
    code, log = pipeline.run()
    assert code != 0, log[-12000:]
    assert "DtvRfiMask" in log and ("mismatch" in log or "malformed" in log), log[
        -12000:
    ]
    assert len(list((tmp_path / "out").glob("dtv_RFImask_*.raw"))) < FRAMES


@pytest.mark.parametrize(
    "settings,reason",
    [
        ({"permanent_mask_freq_ids": [4000, 4000]}, "unique"),
        ({"permanent_mask_freq_ids": [-1]}, "nonnegative"),
        ({"permanent_mask_freq_ids": [True]}, "exact nonnegative"),
        ({"permanent_mask_freq_ids": [4000.5]}, "exact nonnegative"),
        ({"permanent_mask_freq_ids": [1 << 32]}, "exact nonnegative"),
        ({"permanent_mask_freq_ids": [5000]}, "absent"),
        ({"require_fine_freq_ids": [4000]}, "overlap"),
        ({"require_fine_freq_ids": [2623]}, "not evaluated"),
    ],
)
def test_refuse_incomplete_or_ambiguous_threshold_binding(
    tmp_path, runtime, settings, reason
):
    pipeline, *_ = setup_pipeline(tmp_path, runtime, fine_support=True)
    pipeline.config["combine"].update(settings)
    code, log = pipeline.run()
    assert code != 0, log[-12000:]
    assert "DtvRfiMask" in log and reason in log, log[-12000:]


def test_control_frequency_is_never_masked(tmp_path, runtime):
    test_mask_visibility_and_counts_across_ring_wrap(
        tmp_path, runtime, real_detector=False, fine_support=True, control=True
    )


def test_missing_required_decision_masks_the_frame(tmp_path, runtime):
    test_mask_visibility_and_counts_across_ring_wrap(
        tmp_path, runtime, real_detector=False, fine_support=True, missing=True
    )


@pytest.mark.parametrize(
    "settings,reason",
    [
        (
            {"control_freq_ids": [4000]},
            "control and permanent-mask frequencies overlap",
        ),
        ({"control_freq_ids": [5000]}, "absent"),
        ({"missing_decision_action": "keep"}, "missing_decision_action"),
        ({"decision_deadline_s": -1.0}, "decision_deadline_s"),
    ],
)
def test_refuse_bad_control_or_missing_decision_settings(
    tmp_path, runtime, settings, reason
):
    pipeline, *_ = setup_pipeline(tmp_path, runtime, fine_support=True)
    pipeline.config["combine"].update(settings)
    code, log = pipeline.run()
    assert code != 0, log[-12000:]
    assert "DtvRfiMask" in log and reason in log, log[-12000:]


def expected_combined(keep, decision, index, *, decided):
    """The synthetic fine-support case's combined keep mask, with or without its decision."""
    good = keep.copy()
    if decided:
        good &= ~decision.astype(bool)[None, :]
        good[:, 0 if index % 2 else 1] = False
    else:
        good[:, [0, 1]] = False  # every required frequency at the deadline
    good[:, 3] = False  # the permanent mask
    return np.packbits(
        good.reshape(T // 1024, 1024, F).transpose(0, 2, 1), axis=-1, bitorder="little"
    )


@pytest.mark.parametrize("case", ["missing", "stop", "late"])
def test_decision_deadline(tmp_path, runtime, case):
    """Decisions that never come, or come late, release their frames masked at the
    deadline; late ones are discarded; with "stop" a missed deadline stops the run."""
    pipeline, masks, _, decisions = setup_pipeline(tmp_path, runtime, fine_support=True)
    cfg = pipeline.config
    cfg["combine"].update(
        decision_deadline_s=0.3 if case != "late" else 0.1,
        missing_decision_action="stop" if case == "stop" else "mask",
    )
    first_missing = 5
    if case == "late":
        cfg["read_rfi_RFImask"]["frame_period_us"] = 150000
        for name in ("dtv_mask", "dtv_fine_support"):
            cfg[f"read_{name}"]["frame_period_us"] = 400000
    else:
        for name in ("dtv_mask", "dtv_fine_support"):
            for path in sorted((tmp_path / "input").glob(f"{name}_*.raw"))[
                first_missing:
            ]:
                path.unlink()
    code, log = pipeline.run(timeout=120)
    if case == "stop":
        assert code != 0, log[-12000:]
        assert "missed its 0.300 s deadline" in log, log[-12000:]
        return
    assert code == 0, log[-12000:]
    combined = reference.read_raw_frames(
        str(tmp_path / "out/dtv_RFImask_*.raw"), T * F // 8
    )
    assert len(combined) == FRAMES
    missed = []
    for index, frame in enumerate(combined):
        got = frame.payload.reshape(T // 1024, F, 128)
        decided = expected_combined(masks[index], decisions[index], index, decided=True)
        released = expected_combined(
            masks[index], decisions[index], index, decided=False
        )
        if case == "missing":
            np.testing.assert_array_equal(
                got, released if index >= first_missing else decided
            )
        else:
            assert np.array_equal(got, decided) or np.array_equal(got, released), index
        if not np.array_equal(got, decided):
            missed.append(index)
    assert missed, "no frame was released at its deadline"
    assert "deadline: the frame is masked for every required frequency" in log, log[
        -12000:
    ]
    if case == "missing":
        assert "released masked without waiting" in log, log[-12000:]
    else:
        assert "late decision(s) for frames already released" in log, log[-12000:]


def test_finite_threshold_requires_full_rate_support(tmp_path, runtime):
    pipeline, *_ = setup_pipeline(tmp_path, runtime)
    pipeline.config["combine"]["require_fine_freq_ids"] = [2408]
    code, log = pipeline.run()
    assert code != 0, log[-12000:]
    assert "required fine thresholds need fine_support_buf" in log, log[-12000:]


@pytest.mark.parametrize("apply", [False, True])
def test_template_routes_both_correlators_and_rfi_counts(apply):
    env = Environment(loader=FileSystemLoader(str(ROOT / "config/fengine")))
    source = env.get_template("chord.j2").render(dtv_apply_mask=apply)
    cfg = yaml.safe_load(source)
    name = "dtv_RFImask" if apply else "rfi_RFImask"
    for stage in ("run_n2k_correlation", "run_n2k_counts"):
        gpu = cfg[stage]["gpu_0"]
        assert gpu["commands"][0]["rfi_RFImask_name"] == name
        assert f"host_{name}_ringbuffer" in gpu["in_buffers"]
    assert cfg["run_n2_counting"]["count_rfi_mask"]["in_buf"] == f"host_{name}_buffer"
    assert ("run_combine_dtv_mask" in cfg) is apply
    if apply:
        assert (
            cfg["run_combine_dtv_mask"]["fine_support_buf"]
            == "host_dtv_fine_support_buffer"
        )


def test_template_passes_the_same_reject_all_limit_to_both_stages():
    env = Environment(loader=FileSystemLoader(str(ROOT / "config/fengine")))
    cfg = yaml.safe_load(
        env.get_template("chord.j2").render(
            dtv_apply_mask=True,
            dtv_permanent_mask_freq_ids=[4000],
            dtv_require_fine_freq_ids=[2408],
        )
    )
    detector = cfg["run_dtv_detector"]["gpu_0"]["commands"][0]
    combine = cfg["run_combine_dtv_mask"]
    assert (
        detector["permanent_mask_freq_ids"]
        == combine["permanent_mask_freq_ids"]
        == [4000]
    )
    assert combine["require_fine_freq_ids"] == [2408]
    assert detector["dtv_fine_support_name"] == "dtv_fine_support"


def test_chime_mask_visibility_and_counts_across_ring_wrap(tmp_path, runtime):
    """The 16384-sample adapter routes identical samples to both GPU correlators."""
    test_mask_visibility_and_counts_across_ring_wrap(
        tmp_path, runtime, real_detector=False, fine_support=True, num_times=16384
    )


def test_chime_diagnostic_include_has_no_mask_consumer():
    env = Environment(loader=FileSystemLoader(str(ROOT / "config/include")))
    cfg = yaml.safe_load(env.get_template("dtv_chime_upgrade.j2").render())
    gpu = cfg["run_dtv_detector"]["gpu_0"]
    detector = gpu["commands"][0]
    assert detector["receiver_channel_id_namespace"] == "chime_freq_id"
    assert detector["voltage_name"] == "xposed_voltage"
    assert detector["cuda_stream"] == 3
    assert all(command["cuda_stream"] == 4 for command in gpu["commands"][1:])
    assert gpu["num_times"] == "samples_per_data_set"
    assert set(cfg) == {
        "host_dtv_mask_buffer",
        "host_dtv_powers_buffer",
        "host_dtv_fine_support_buffer",
        "run_dtv_detector",
        "write_dtv",
    }
    assert all(
        row["kotekan_stage"] == "rawFileWrite" for row in cfg["write_dtv"].values()
    )


@pytest.mark.parametrize("reject_finite", [False, True])
def test_chime_detector_mask_visibility_and_counts_across_ring_wrap(
    tmp_path, runtime, reject_finite
):
    """The K128 detector, support, adapter and both correlators agree."""
    import json

    bundle = json.loads((runtime[1] / "pilot_profiles.json").read_text())
    if bundle["profiles"][0]["weight_bank_nbytes"] != 3 * 128:
        pytest.skip("requires the K128 detector and matching bundle")
    test_mask_visibility_and_counts_across_ring_wrap(
        tmp_path,
        runtime,
        real_detector=True,
        fine_support=True,
        num_times=16384,
        reject_finite=reject_finite,
    )


def test_template_passes_control_failure_and_bad_input_settings():
    env = Environment(loader=FileSystemLoader(str(ROOT / "config/fengine")))
    default = yaml.safe_load(env.get_template("chord.j2").render(dtv_apply_mask=True))
    detector = default["run_dtv_detector"]["gpu_0"]["commands"][0]
    combine = default["run_combine_dtv_mask"]
    assert detector["control_freq_ids"] == combine["control_freq_ids"] == []
    assert (
        detector["runtime_failure_mode"] == combine["missing_decision_action"] == "stop"
    )
    assert combine["decision_deadline_s"] == 0 and "bf_mask_name" not in detector
    assert (
        "host_bf_mask_ringbuffer"
        not in default["run_dtv_detector"]["gpu_0"]["in_buffers"]
    )
    cfg = yaml.safe_load(
        env.get_template("chord.j2").render(
            dtv_apply_mask=True,
            dtv_control_freq_ids=[4000],
            dtv_exclude_bad_inputs=True,
            dtv_runtime_failure_mode="degrade",
            dtv_missing_decision_action="mask",
            dtv_decision_deadline_s=0.05,
        )
    )
    gpu = cfg["run_dtv_detector"]["gpu_0"]
    detector, combine = gpu["commands"][0], cfg["run_combine_dtv_mask"]
    assert detector["control_freq_ids"] == combine["control_freq_ids"] == [4000]
    assert detector["runtime_failure_mode"] == "degrade"
    assert combine["missing_decision_action"] == "mask"
    assert combine["decision_deadline_s"] == 0.05
    assert detector["bf_mask_name"] == "bf_mask"
    assert gpu["in_buffers"]["host_bf_mask_ringbuffer"] == "host_bf_mask_ringbuffer"
    assert "bf_mask_lifetime_in_samples" in cfg and "host_bf_mask_ringbuffer" in cfg


def packed(good):
    return np.packbits(
        good.reshape(T // 1024, 1024, F).transpose(0, 2, 1), axis=-1, bitorder="little"
    )


@pytest.mark.parametrize("case", ["not_evaluated", "deadline"])
def test_missing_decisions_count_as_lost_not_rfi(tmp_path, runtime, case):
    """Frames without a decision leave the visibilities, count as lost samples in the
    packet-loss counts N2Accumulate reads, and stay out of RfiMaskSum's RFI count."""
    pipeline, masks, _, decisions = setup_pipeline(
        tmp_path, runtime, fine_support=True, plcounts=True
    )
    freq_ids = pipeline.generator["manual_freq_ids"]
    combine = pipeline.config["combine"]
    combine["missing_decision_action"] = "mask"
    first_missing = 5
    if case == "not_evaluated":
        # Frequency 2's support rows are (-1, -1).
        combine["require_fine_freq_ids"] = combine["require_fine_freq_ids"] + [
            freq_ids[2]
        ]
    else:
        combine["decision_deadline_s"] = 0.3
        for name in ("dtv_mask", "dtv_fine_support"):
            for path in sorted((tmp_path / "input").glob(f"{name}_*.raw"))[
                first_missing:
            ]:
                path.unlink()
    code, log = pipeline.run(timeout=120)
    assert code == 0, log[-12000:]

    def read(name, size):
        frames = reference.read_raw_frames(str(tmp_path / f"out/{name}_*.raw"), size)
        assert len(frames) == FRAMES
        return frames

    combined = read("dtv_RFImask", T * F // 8)
    rficounts = read("rficounts", F * 4)
    lost = read("dtv_plcounts", F * 4)
    for index in range(FRAMES):
        rfi_good = masks[index].copy()
        missing = np.zeros(F, dtype=bool)
        if case == "not_evaluated" or index < first_missing:
            rfi_good &= ~decisions[index].astype(bool)[None, :]
            rfi_good[:, 0 if index % 2 else 1] = False  # invalid rank support is RFI
            missing[2] = case == "not_evaluated"
        else:
            missing[[0, 1]] = True  # every required frequency at the deadline
        rfi_good[:, 3] = False  # the permanent mask
        good = rfi_good & ~missing[None, :]
        np.testing.assert_array_equal(
            combined[index].payload.reshape(T // 1024, F, 128), packed(good)
        )
        np.testing.assert_array_equal(
            rficounts[index].payload.view("<i4"), (~rfi_good).sum(axis=0)
        )
        want = np.array([10 * index + f for f in range(F)], dtype="<i4")
        want[missing] = T
        np.testing.assert_array_equal(lost[index].payload.view("<i4"), want)
        assert lost[index].fpga_seq_num == combined[index].fpga_seq_num


def add_decoupled_detector(pipeline):
    """Give the detector its own voltage copy, detached by bufferCopy when it stays full."""
    cfg = pipeline.config
    # Two frames of copy and two of ring, so the eight-frame test overruns them.
    cfg["host_dtv_voltage_buffer"] = copy.deepcopy(cfg["host_voltage_buffer"])
    cfg["host_dtv_voltage_buffer"]["num_frames"] = 2
    cfg["host_dtv_voltage_ringbuffer"] = copy.deepcopy(cfg["host_voltage_ringbuffer"])
    cfg["host_dtv_voltage_ringbuffer"][
        "ring_buffer_size"
    ] = "2 * num_times * num_local_freq * num_polarizations * num_dishes"
    cfg["copy_dtv_voltage"] = dict(
        kotekan_stage="bufferCopy",
        in_buf="host_voltage_buffer",
        out_bufs=["host_dtv_voltage_buffer"],
        detach_timeout_s=0.2,
    )
    sender = copy.deepcopy(cfg["run_send_voltage"])
    sender["gpu_0"]["buffer_depth"] = 2  # one pipeline instance per copy frame
    sender["gpu_0"]["in_buffers"] = {
        "host_dtv_voltage_buffer": "host_dtv_voltage_buffer"
    }
    sender["gpu_0"]["out_buffers"] = {
        "host_dtv_voltage_ringbuffer": "host_dtv_voltage_ringbuffer"
    }
    sender["gpu_0"]["commands"][0].update(
        in_buf="host_dtv_voltage_buffer",
        signal_buf="host_dtv_voltage_ringbuffer",
        gpu_mem_output="dtv_voltage_buffer",
        ring_buffer_size="2 * input_size",
    )
    cfg["send_dtv_voltage"] = sender
    gpu = cfg["run_dtv_detector"]["gpu_0"]
    gpu["in_buffers"] = {"host_dtv_voltage_ringbuffer": "host_dtv_voltage_ringbuffer"}
    gpu["commands"][0]["voltage_name"] = "dtv_voltage"
    gpu["buffer_depth"] = 2  # the depth of its voltage ring


@pytest.mark.parametrize("decoupled", [True, False])
def test_hung_detector_cannot_stall_the_correlator(tmp_path, runtime, decoupled):
    """A detector that stops returning: with its own detachable voltage copy the
    correlator keeps flowing and the frames are released masked at the deadline;
    sharing the correlator's ring, the correlator waits for the detector."""
    hang = 10.0
    pipeline, masks, _, _ = setup_pipeline(
        tmp_path, runtime, real_detector=True, fine_support=True
    )
    cfg = pipeline.config
    freq_ids = pipeline.generator["manual_freq_ids"]
    if decoupled:
        add_decoupled_detector(pipeline)
    cfg["run_dtv_detector"]["gpu_0"]["commands"][0].update(
        fault_hang_frame=2, fault_hang_seconds=hang
    )
    cfg["combine"].update(missing_decision_action="mask", decision_deadline_s=0.5)
    # A stalled detector writes fewer decisions before the finite input ends; the
    # correlator outputs end the run.
    for name in ("dump_dtv_mask", "dump_dtv_powers", "dump_dtv_fine_support"):
        cfg[name].pop("exit_after_n_files", None)
    code, log = pipeline.run(timeout=120)
    assert code == 0, log[-12000:]
    assert "blocking frame 2" in log, log[-12000:]
    vis = sorted((tmp_path / "out").glob("n2k_correlation_*.raw"))
    assert len(vis) == FRAMES
    # The detector wrote its first frame before it blocked.
    first = sorted((tmp_path / "out").glob("dtv_mask_*.raw"))[0].stat().st_mtime
    spread = vis[-1].stat().st_mtime - first
    combined = reference.read_raw_frames(
        str(tmp_path / "out/dtv_RFImask_*.raw"), T * F // 8
    )
    released = []
    for index, frame in enumerate(combined):
        good = masks[index].copy()
        good[:, [0, 2]] = False  # the permanent mask and the required frequency
        if np.array_equal(frame.payload.reshape(T // 1024, F, 128), packed(good)):
            released.append(index)
    if decoupled:
        assert "DETACHED" in log, log[-12000:]
        assert (
            spread < hang / 2
        ), f"last correlator frame {spread:.1f} s after the first decision"
        assert len(released) >= 3 and released[-1] == FRAMES - 1, released
    else:
        assert (
            spread > hang * 0.8
        ), f"last correlator frame {spread:.1f} s after the first decision"
    assert freq_ids  # the released expectation covers the configured frequencies


@pytest.mark.parametrize("lost", [False, True])
def test_template_routes_lost_samples_and_the_decoupled_detector(lost):
    env = Environment(loader=FileSystemLoader(str(ROOT / "config/fengine")))
    settings = dict(dtv_apply_mask=True)
    if lost:
        settings.update(
            dtv_missing_decision_action="mask",
            dtv_decision_deadline_s=0.05,
            dtv_decouple_detector=True,
        )
    cfg = yaml.safe_load(env.get_template("chord.j2").render(**settings))
    combine = cfg["run_combine_dtv_mask"]
    gpu = cfg["run_dtv_detector"]["gpu_0"]
    rfi_count = cfg["run_n2_counting"]["count_rfi_mask"]["in_buf"]
    plcounts = cfg["run_N2_accumulate"]["in_plcounts_buf"]
    if not lost:
        assert rfi_count == "host_dtv_RFImask_buffer"
        assert plcounts == "host_n2k_plcounts_buffer"
        assert "plcounts_in_buf" not in combine and "run_copy_dtv_voltage" not in cfg
        assert gpu["commands"][0]["voltage_name"] == "voltage"
        return
    assert combine["plcounts_in_buf"] == "host_n2k_plcounts_buffer"
    assert combine["plcounts_out_buf"] == plcounts == "host_dtv_plcounts_buffer"
    assert combine["rfi_count_buf"] == rfi_count == "host_dtv_rfi_count_buffer"
    assert cfg["run_copy_dtv_voltage"]["in_buf"] == "host_voltage_buffer"
    assert cfg["run_copy_dtv_voltage"]["detach_timeout_s"] == 1.0
    assert gpu["in_buffers"] == {
        "host_dtv_voltage_ringbuffer": "host_dtv_voltage_ringbuffer"
    }
    assert gpu["commands"][0]["voltage_name"] == "dtv_voltage"
