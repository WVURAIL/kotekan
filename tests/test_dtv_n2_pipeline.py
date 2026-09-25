"""CUDA detector/mask/correlators and native N2 accumulation in one process.

Synthetic calibration and eight repeating voltage seeds test arithmetic and
scheduling, not independent noise, telescope calibration, or live throughput.
"""
import ctypes
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from kotekan import n2buffer
from test_dtv_rfi_mask import F, FRAMES, T, setup_pipeline, triangles
from test_pilotproxy_integration import reference, runtime

pytestmark = pytest.mark.serial
E = 128
BOOT_NS = 1767287750500000000
FREQ = [2408, 1600, 2623, 4000]


class Header(ctypes.Structure):
    """Fixed prefix of the native chordMetadataFormat used by rawFileRead."""

    _fields_ = [
        ("max_dim", ctypes.c_int32),
        ("max_name", ctypes.c_int32),
        ("max_dimname", ctypes.c_int32),
        ("max_freq", ctypes.c_int32),
        ("max_stream_ids", ctypes.c_int32),
        ("max_rfi_thresholds", ctypes.c_int32),
        ("frame_counter", ctypes.c_int32),
        ("fpga_seq_num", ctypes.c_int64),
        ("period", ctypes.c_int32),
        ("name", ctypes.c_char * 24),
        ("type", ctypes.c_int32),
        ("dims", ctypes.c_int32),
        ("dim", ctypes.c_int32 * 10),
        ("dim_name", (ctypes.c_char * 24) * 10),
        ("scaling", ctypes.c_int64 * 10),
        ("stride", ctypes.c_int64 * 10),
        ("offset", ctypes.c_int64),
    ]


def rewrite(
    raw,
    seq,
    period,
    *,
    name=None,
    dtype=None,
    shape=None,
    dims=None,
    scales=None,
    payload=None,
):
    if len(raw) < 4:
        raise ValueError("truncated metadata length")
    size = struct.unpack_from("<I", raw)[0]
    if size != reference.CHORD_METADATA_SIZE or len(raw) < size + 4:
        raise ValueError("unsupported or truncated CHORD metadata")
    data = bytearray(raw[: size + 4])
    h = Header.from_buffer(data, 4)
    limits = (
        h.max_dim, h.max_name, h.max_dimname, h.max_freq,
        h.max_stream_ids, h.max_rfi_thresholds,
    )
    if limits != reference.CHORD_METADATA_LIMITS:
        raise ValueError("unsupported CHORD metadata layout")
    h.fpga_seq_num, h.period = seq, period
    if name is not None:
        h.name, h.type, h.dims = name.encode(), dtype, len(shape)
        stride = 1
        for i in reversed(range(len(shape))):
            h.dim[i], h.scaling[i], h.stride[i] = shape[i], scales[i], stride
            h.dim_name[i].value = dims[i].encode()
            stride *= shape[i]
        h.offset = 0
    del h
    data.extend(raw[size + 4 :] if payload is None else payload)
    return data


def ndarray(dtype, quantity, shape, dims, scales):
    return dict(
        kotekan_buffer="ndarray",
        num_frames=4,
        metadata_pool="main_pool",
        value_type=dtype,
        quantity_name=quantity,
        extents=shape,
        dimnames=dims,
        dimscalings=scales,
    )


def composed(tmp_path, runtime, *, real_detector=True, subintegrations=4, fault=None):
    pipeline, masks, packets, decisions = setup_pipeline(
        tmp_path, runtime, real_detector=real_detector
    )
    cfg = pipeline.config
    # Keep the original seed files, and place the longer replay in its own directory.
    source = tmp_path / "input"
    target = tmp_path / "replay"
    target.mkdir()
    seeds = {
        name: [p.read_bytes() for p in sorted(source.glob(f"{name}_*.raw"))]
        for name in ("voltage", "rfi_RFImask", "pl_expanded_mask", "dtv_mask")
    }
    for i in range(FRAMES):
        pl = packets[i]
        pl[T // 2 :, :, 0] = False
        pl[: T // 2, :, 1] = False
        if i % 2:
            pl[:, :, 4] = False
        z = np.frombuffer(
            seeds["voltage"][i],
            dtype=np.uint8,
            offset=struct.unpack_from("<I", seeds["voltage"][i])[0] + 4,
        ).copy()
        z = z.reshape(T, F, E)
        z[~np.repeat(pl, 8, axis=2)] = 0x88
        seeds["voltage"][i] = rewrite(seeds["voltage"][i], 0, 1, payload=z.tobytes())
        packed = np.packbits(
            pl.reshape(128, 64, F, 2, 8).transpose(0, 2, 3, 4, 1),
            axis=-1,
            bitorder="little",
        )
        seeds["pl_expanded_mask"][i] = rewrite(
            seeds["pl_expanded_mask"][i], 0, 64, payload=packed.tobytes()
        )
    nframes = 2 * subintegrations
    for i in range(nframes):
        seq = (subintegrations + i) * T
        for name, period in (
            ("voltage", 1),
            ("rfi_RFImask", 1024),
            ("pl_expanded_mask", 64),
            ("dtv_mask", T),
        ):
            (target / f"{name}_{i:07d}.raw").write_bytes(
                rewrite(seeds[name][i % FRAMES], seq, period)
            )
        # This scalar diagnostic is unavailable in per-product output; joint
        # support comes from the real packet-mask correlator, never this zero.
        (target / f"plcounts_{i:07d}.raw").write_bytes(
            rewrite(
                seeds["voltage"][0],
                seq,
                T,
                name="pl_lost_counts_scalar",
                dtype=11,
                shape=[1, F],
                dims=["Tc", "F"],
                scales=[T, 1],
                payload=np.zeros(F, dtype="<i4").tobytes(),
            )
        )
        sk = np.zeros((1, F, 3), dtype="<f4")
        sk[..., 0] = 1
        sk[..., 2] = 1
        if i % FRAMES == 6:
            sk[0, 3, 0] = 10
        (target / f"sk_{i:07d}.raw").write_bytes(
            rewrite(
                seeds["voltage"][0],
                seq + (1 if fault == "time" and i == 4 else 0),
                T,
                name="SKtilde",
                dtype=14,
                shape=[1, F, 3],
                dims=["Trfi", "F", "SK"],
                scales=[T, 1, 1],
                payload=sk.tobytes(),
            )
        )
    for name in seeds:
        if f"read_{name}" in cfg:
            cfg[f"read_{name}"]["base_dir"] = "replay"
    # All writers participate in the shared completion barrier, including the
    # native writer below; neither intermediate nor final products may be truncated.
    for stage in cfg.values():
        if isinstance(stage, dict) and stage.get("kotekan_stage") == "rawFileWrite":
            stage["exit_after_n_files"] = nframes
    cfg.update(
        samples_per_data_set=T,
        sub_integration_ntime=T,
        num_elements=E,
        num_ev=0,
        num_workers=1,
    )
    cfg["N2_pool"] = dict(kotekan_metadata_pool="N2Metadata", num_metadata_objects=64)
    cfg["gps_time"] = {"frame0_nano": BOOT_NS}
    cfg["eop"] = dict(
        kotekan_update_endpoint="json",
        earth_orientation_parameter_table=[
            dict(t_inst_ns=BOOT_NS + delta, delta_UT1_inst=0.0, xp_as=0.0, yp_as=0.0)
            for delta in [-3600_000_000_000, 3600_000_000_000]
        ],
    )
    cfg["telescope"] = dict(
        name="CHORDTelescope",
        sampling_rate_MHz=3200.0,
        fft_length=16384,
        nyquist_zone=1,
        dish_grid_size_x=12,
        dish_grid_size_y=6,
        dish_inputs=[],
        origin_itrs_lat_deg=50.0,
        origin_itrs_lon_deg=-120.0,
        eop_updatable_config="/eop",
        require_eop=True,
        fatal_eop_out_of_range=True,
    )
    cfg["host_n2k_correlation_buffer"] = ndarray(
        "int32",
        "n2k_correlation",
        [1, F, 36, 16, 16, 2],
        ["Tc", "F", "DPhi", "DPlo1", "DPlo2", "C"],
        [T, 1, 16, 1, 1, 1],
    )
    cfg["host_n2k_counts_buffer"] = ndarray(
        "int32",
        "n2k_counts",
        [1, F, 3, 8, 8],
        ["Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"],
        [T, 1, 64, 8, 8],
    )
    cfg["host_plcounts_buffer"] = ndarray(
        "int32", "pl_lost_counts_scalar", [1, F], ["Tc", "F"], [T, 1]
    )
    cfg["host_sk_buffer"] = ndarray(
        "float32", "SKtilde", [1, F, 3], ["Trfi", "F", "SK"], [T, 1, 1]
    )
    cfg["host_framemask_buffer"] = ndarray(
        "uint8", "RFIFrameMask", [1, F], ["Tc", "F"], [T, 1]
    )
    for name in ["plcounts", "sk"]:
        cfg[f"read_{name}"] = dict(
            kotekan_stage="rawFileRead",
            strict_framing=True,
            buf=f"host_{name}_buffer",
            base_dir="replay",
            file_name=name,
            file_ext="raw",
            prefix_hostname=False,
            end_interrupt=False,
        )
    cfg["frame_enabled"] = dict(
        kotekan_update_endpoint="json", valid_at_time_ns=BOOT_NS, enabled=True
    )
    cfg["frame_thresholds"] = dict(
        kotekan_update_endpoint="json",
        valid_at_time_ns=BOOT_NS,
        thresholds=[dict(threshold=3.0, fraction=0.1)],
    )
    cfg["frame_mask"] = dict(
        kotekan_stage="RfiFrameMask",
        in_buf="host_sk_buffer",
        out_buf="host_framemask_buffer",
        rfi_downsampling_factor=T,
        enabled_updatable_config="/frame_enabled",
        thresholds_updatable_config="/frame_thresholds",
    )
    cfg["native_buffer"] = dict(
        kotekan_buffer="N2",
        n2_layout="FullUpperTri",
        support_mode="scalar" if fault == "support" else "per_product_v1",
        metadata_pool="N2_pool",
        num_frames=4 * F,
    )
    cfg["accumulate"] = dict(
        kotekan_stage="N2Accumulate",
        num_freq_per_n2k_frame=F,
        packet_loss_is_scalar=False,
        bin_in_ERA=False,
        num_subintegrations_per_bin=subintegrations,
        variance_mode="EvenOddPosDef",
        do_fringestop=False,
        input_order="CHORDBeamformer",
        output_order="CHORDBeamformer",
        in_buf="host_n2k_correlation_buffer",
        in_counts_buf="host_n2k_counts_buffer",
        in_rficounts_buf="host_rficounts_buffer",
        in_plcounts_buf="host_plcounts_buffer",
        in_rfiframemask_buf="host_framemask_buffer",
        out_buf="native_buffer",
    )
    cfg["dump_native"] = dict(
        kotekan_stage="rawFileWrite",
        in_buf="native_buffer",
        file_name="native",
        file_ext="dump",
        prefix_hostname=False,
        num_frames_per_file=1,
        exit_after_n_files=2 * F,
    )
    return pipeline, masks, packets, seeds, decisions


def verify(
    tmp_path, pipeline, masks, packets, seeds, decisions, subintegrations, real_detector
):
    nframes = 2 * subintegrations

    def read(name, size):
        data = reference.read_raw_frames(str(tmp_path / "out" / f"{name}_*.raw"), size)
        assert len(data) == nframes
        return data

    masks_out = read("dtv_RFImask", T * F // 8)
    vis = read("n2k_correlation", F * 36 * 16 * 16 * 8)
    counts = read("n2k_counts", F * 3 * 8 * 8 * 4)
    rfi = read("rficounts", F * 4)
    if real_detector:
        actual_decisions = read("dtv_mask", F)
        actual_powers = read("dtv_powers", F * 3 * 8)
        expected = [
            reference.expected_products(
                np.frombuffer(
                    raw, dtype=np.uint8, offset=struct.unpack_from("<I", raw)[0] + 4
                ),
                FREQ,
                pipeline.bundle,
                (tmp_path / "bundle/weights.bin").read_bytes(),
                num_dishes=64,
                decision_mode="auto",
            )
            for raw in seeds["voltage"]
        ]
        decisions = [x[0] for x in expected]
        for i in range(nframes):
            np.testing.assert_array_equal(
                actual_decisions[i].payload, decisions[i % FRAMES]
            )
            np.testing.assert_array_equal(
                actual_powers[i].payload.view("<u8").reshape(F, 3),
                expected[i % FRAMES][1],
            )
    sums, joint = [], []
    rows, cols = np.triu_indices(E)
    for i, raw in enumerate(seeds["voltage"]):
        good = masks[i] & ~np.asarray(decisions[i], bool)[None, :]
        data = np.frombuffer(
            raw, dtype=np.uint8, offset=struct.unpack_from("<I", raw)[0] + 4
        )
        data = data.reshape(T, F, E).astype(np.int16)
        z = ((data >> 4) - 8 + 1j * ((data & 15) - 8)).astype(np.complex64)
        z *= good[..., None]
        # Integer-valued sums stay below the exact float32 integer limit.
        matrix = np.stack([z[:, f].T @ z[:, f].conj() for f in range(F)])
        valid = (packets[i] & good[..., None]).astype(np.float64)
        count = np.stack([valid[:, f].T @ valid[:, f] for f in range(F)]).astype(
            np.uint64
        )
        sums.append(matrix[:, rows, cols].astype(np.complex128))
        joint.append(count[:, rows // 8, cols // 8])
        want_vis, want_count = triangles(matrix, 16), triangles(count, 8)
        for j in range(i, nframes, FRAMES):
            np.testing.assert_array_equal(
                vis[j].payload.view("<i4").reshape(F, 36, 16, 16, 2)[..., 0],
                want_vis.real,
            )
            np.testing.assert_array_equal(
                vis[j].payload.view("<i4").reshape(F, 36, 16, 16, 2)[..., 1],
                want_vis.imag,
            )
            np.testing.assert_array_equal(
                counts[j].payload.view("<i4").reshape(F, 3, 8, 8), want_count
            )
            np.testing.assert_array_equal(rfi[j].payload.view("<i4"), (~good).sum(0))
            np.testing.assert_array_equal(
                masks_out[j].payload.reshape(8, F, 128),
                np.packbits(
                    good.reshape(8, 1024, F).transpose(0, 2, 1),
                    axis=-1,
                    bitorder="little",
                ),
            )
            for product, period in [(vis, T), (counts, T), (masks_out, 1024)]:
                assert product[j].fpga_seq_num == (subintegrations + j) * T
                assert product[j].time_downsampling_fpga == period
    actual = n2buffer.N2Buffer.load_files(
        str(tmp_path / "out/native_*.dump"),
        num_elements=E,
        num_prod=E * (E + 1) // 2,
        num_ev=0,
        support_mode="per_product_v1",
    )
    assert len(actual) == 2 * F
    actual = {
        (int(x.metadata.abs_time_idx), int(x.metadata.freq_id)): x for x in actual
    }
    receipts = []
    for b in range(2):
        n = np.zeros((F, len(rows)), dtype=np.uint64)
        k = np.zeros_like(n)
        total = np.zeros(n.shape, dtype=np.complex128)
        q = np.zeros(n.shape)
        for j in range(b * subintegrations, (b + 1) * subintegrations, 2):
            a, z = j % FRAMES, (j + 1) % FRAMES
            admit = np.ones(F, bool)
            if 6 in [a, z]:
                admit[3] = False
            n0, n1 = joint[a], joint[z]
            valid = (n0 > 0) & (n1 > 0) & admit[:, None]
            n += (n0 + n1) * admit[:, None]
            total += (sums[a] + sums[z]) * admit[:, None]
            k += valid
            diff = np.divide(
                sums[a], n0, out=np.zeros_like(total), where=n0 > 0
            ) - np.divide(sums[z], n1, out=np.zeros_like(total), where=n1 > 0)
            factor = np.divide(
                n0.astype(float) * n1,
                n0 + n1,
                out=np.zeros(q.shape),
                where=(n0 + n1) > 0,
            )
            q += np.where(valid, factor * abs(diff) ** 2, 0)
        mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
        weight = np.divide(
            n.astype(float) * k, q, out=np.zeros_like(q), where=(q > 0) & (k > 0)
        )
        for f, fid in enumerate(FREQ):
            got = actual[(b + 1, fid)]
            np.testing.assert_array_equal(got.valid_fpga_ticks, n[f])
            np.testing.assert_allclose(got.vis, mean[f], rtol=2e-6, atol=1e-7)
            np.testing.assert_allclose(got.weight, weight[f], rtol=4e-6, atol=1e-7)
            assert got.metadata.fpga_start_tick == (b + 1) * subintegrations * T
            assert got.metadata.frame_length_fpga_ticks == subintegrations * T
            assert (
                got.metadata.frame_start_time_ns
                == BOOT_NS + (b + 1) * subintegrations * T * 5120
            )
            for key in [
                "n_valid_fpga_ticks",
                "n_pl_fpga_ticks",
                "n_rfi_fpga_ticks",
                "n_rfi_only_fpga_ticks",
            ]:
                assert getattr(got.metadata, key) == 0
            assert np.all(got.vis[n[f] == 0] == 0)
            assert np.all(got.weight[n[f] == 0] == 0)
            assert np.all(got.weight[(n[f] > 0) & (k[f] == 0)] == 0)
            assert np.isfinite(got.vis).all() and np.isfinite(got.weight).all()
            receipts.append(
                dict(
                    bin=b + 1,
                    frequency=fid,
                    max_visibility_error=float(np.max(abs(got.vis - mean[f]))),
                    max_weight_error=float(np.max(abs(got.weight - weight[f]))),
                    zero_support=int(np.sum(n[f] == 0)),
                    positive_support_without_pair=int(np.sum((n[f] > 0) & (k[f] == 0))),
                    min_joint_ticks=int(n[f].min()),
                    max_joint_ticks=int(n[f].max()),
                )
            )
    assert any(x["zero_support"] for x in receipts)
    assert any(x["positive_support_without_pair"] for x in receipts)
    receipt = dict(
        real_detector=real_detector,
        frames=nframes,
        unique_voltage_seeds=FRAMES,
        subintegrations_per_bin=subintegrations,
        integration_seconds=subintegrations * T * 5.12e-6,
        one_process=True,
        support_mode="per_product_v1",
        fringestopping=False,
        independent_noise_validation=False,
        physical_calibration=False,
        shared_gpu_timing_acceptance=False,
        outputs=receipts,
    )
    (tmp_path / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


@pytest.mark.parametrize("real_detector", [False, True])
def test_composed_cuda_native_accumulation(tmp_path, runtime, real_detector):
    pipeline, *inputs = composed(tmp_path, runtime, real_detector=real_detector)
    code, log = pipeline.run(timeout=120)
    assert code == 0, log[-16000:]
    verify(tmp_path, pipeline, *inputs, subintegrations=4, real_detector=real_detector)


def test_composed_nominal_ten_seconds(tmp_path, runtime):
    if os.environ.get("PILOTPROXY_N2_TEN_SECONDS") != "1":
        pytest.skip("set PILOTPROXY_N2_TEN_SECONDS=1 for the 480-block CUDA replay")
    pipeline, *inputs = composed(tmp_path, runtime, subintegrations=240)
    code, log = pipeline.run(timeout=300)
    assert code == 0, log[-16000:]
    verify(tmp_path, pipeline, *inputs, subintegrations=240, real_detector=True)


@pytest.mark.parametrize("fault", ["time", "support"])
def test_composed_refuses_invalid_accumulator_contract(tmp_path, runtime, fault):
    pipeline, *_ = composed(tmp_path, runtime, fault=fault)
    code, log = pipeline.run(timeout=60)
    assert code != 0, log[-16000:]
    if fault == "time":
        assert "Correlation buffer host_n2k_correlation_buffer" in log
        assert "has lost synchronization with RFIFrameMask" in log
    else:
        assert "N2Accumulate support_mode must match packet_loss_is_scalar" in log, log[
            -16000:
        ]
    assert len(list((tmp_path / "out").glob("native_*.dump"))) < 2 * F
