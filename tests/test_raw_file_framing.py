"""Real CPU rawFileWrite -> rawFileRead round trips, including repeated headers."""
import ctypes
import hashlib
import json
import os
import struct
from pathlib import Path

import pytest
from kotekan.n2buffer import N2Buffer, N2Metadata

from kotekan import runner

pytestmark = pytest.mark.serial


def specification(kind):
    if kind == "standard":
        return (
            64,
            {
                "kotekan_buffer": "standard",
                "metadata_pool": "main_pool",
                "frame_size": 64,
                "num_frames": 3,
            },
        )
    mode = "per_product_v1" if kind == "n2-product" else "scalar"
    size = N2Buffer.calculate_layout(4, 10, 0, support_mode=mode)["size"]
    return (
        size,
        {
            "kotekan_buffer": "N2",
            "metadata_pool": "N2_pool",
            "n2_layout": "FullUpperTri",
            "support_mode": mode,
            "num_frames": 3,
        },
    )


def records(kind, count):
    size, _ = specification(kind)
    result = []
    for i in range(count):
        payload = bytes((k * 11 + i * 37) % 256 for k in range(size))
        metadata = b""
        if kind != "standard":
            meta = N2Metadata()
            meta.abs_time_idx = i + 100
            meta.freq_id = i + 600
            meta.fpga_start_tick = i * 128
            meta.frame_length_fpga_ticks = 128
            metadata = bytes(meta)
            assert len(metadata) == ctypes.sizeof(N2Metadata)
        result.append(struct.pack("<I", len(metadata)) + metadata + payload)
    return result


def transfer(
    source, destination, kind, *, frames_per_file, output_files, failure=False
):
    destination.mkdir(exist_ok=True)
    _, buffer = specification(kind)
    stages = {
        "read": {
            "kotekan_stage": "rawFileRead",
            "buf": "transport",
            "base_dir": str(source),
            "file_name": "record",
            "file_ext": "raw",
            "prefix_hostname": False,
            "end_interrupt": True,
        },
        "write": {
            "kotekan_stage": "rawFileWrite",
            "in_buf": "transport",
            "base_dir": str(destination),
            "file_name": "record",
            "file_ext": "raw",
            "prefix_hostname": False,
            "num_frames_per_file": frames_per_file,
            "exit_after_n_files": output_files,
        },
    }
    config = {
        "buffer_depth": 3,
        "num_elements": 4,
        "num_dishes": 2,
        "num_polarizations": 2,
        "num_ev": 0,
        "log_level": "INFO",
    }
    task = runner.KotekanRunner(
        buffers={"transport": buffer},
        stages=stages,
        config=config,
        expect_failure=failure,
        timeout=15,
    )
    task.run()
    return task


def evidence(label, record):
    destination = os.getenv("RAW_FILE_FRAMING_EVIDENCE_DIR")
    if destination:
        p = Path(destination)
        p.mkdir(parents=True, exist_ok=True)
        (p / (label + ".json")).write_text(json.dumps(record, indent=2) + "\n")


@pytest.mark.parametrize("kind", ["standard", "n2-scalar", "n2-product"])
@pytest.mark.parametrize("count", [1, 3])
def test_actual_writer_reader_roundtrip(tmp_path, kind, count):
    source = tmp_path / "source"
    source.mkdir()
    expected = records(kind, count)
    for i, record in enumerate(expected):
        (source / f"record_{i:07d}.raw").write_bytes(record)
    packed = tmp_path / "packed"
    transfer(source, packed, kind, frames_per_file=count, output_files=1)
    written = (packed / "record_0000000.raw").read_bytes()
    # Check actual writer output independently, including every metadata-size word.
    assert written == b"".join(expected)
    unpacked = tmp_path / "unpacked"
    transfer(packed, unpacked, kind, frames_per_file=1, output_files=count)
    actual = [p.read_bytes() for p in sorted(unpacked.glob("record_*.raw"))]
    assert actual == expected
    evidence(
        f"roundtrip-{kind}-{count}",
        {
            "success": True,
            "frames": count,
            "kind": kind,
            "writer_bytes": len(written),
            "per_record_metadata_header": True,
            "exact_roundtrip_bytes": True,
            "writer_sha256": hashlib.sha256(written).hexdigest(),
        },
    )


@pytest.mark.parametrize(
    "fault,diagnostic",
    [
        ("short-header", "missing size header"),
        ("truncated", "raw file payload"),
        ("trailing-extension", "raw file payload"),
        ("changed-header", "metadata size changed between records"),
        ("metadata-type-size", "serialized metadata size does not match"),
        ("n2-no-metadata", "N2 frames require metadata"),
        ("wrong-support-mode", "raw file payload"),
        ("large-sparse-size", "raw file payload"),
    ],
)
def test_invalid_raw_framing_refused(tmp_path, fault, diagnostic):
    source = tmp_path / "source"
    source.mkdir()
    size, _ = specification("n2-scalar")
    kind = "n2-scalar"
    packed = b"".join(records(kind, 3))
    if fault == "short-header":
        packed = b"\0\0\0"
    elif fault == "truncated":
        packed = packed[:-1]
    elif fault == "trailing-extension":
        packed += b"x"
    elif fault == "changed-header":
        first = len(records(kind, 1)[0])
        data = bytearray(packed)
        struct.pack_into("<I", data, first, ctypes.sizeof(N2Metadata) + 1)
        packed = bytes(data)
    elif fault == "metadata-type-size":
        packed = struct.pack("<I", 1) + b"x" + bytes(size)
    elif fault == "n2-no-metadata":
        packed = struct.pack("<I", 0) + bytes(size)
    elif fault == "wrong-support-mode":
        packed = records("n2-product", 1)[0]
    elif fault == "large-sparse-size":
        kind = "standard"
        packed = struct.pack("<I", 0) + bytes(64)
    file = source / "record_0000000.raw"
    file.write_bytes(packed)
    if fault == "large-sparse-size":
        # Constant physical allocation; upper file-size bits must affect framing.
        with file.open("r+b") as f:
            f.truncate((1 << 32) + len(packed))
        assert file.stat().st_size > (1 << 32)
        assert file.stat().st_size % len(packed) != 0
    task = transfer(
        source,
        tmp_path / "output",
        kind,
        frames_per_file=1,
        output_files=3,
        failure=True,
    )
    assert task.return_code != 0
    assert diagnostic in task.output
    evidence(
        f"refusal-{fault}",
        {
            "success": True,
            "failure_observed": True,
            "file_bytes": file.stat().st_size,
            "diagnostic": diagnostic,
            "return_code": task.return_code,
        },
    )
