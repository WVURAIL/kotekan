"""CPU per-product fringe rotation against an independent zenith/ENU oracle.

Scope: deterministic CHIMEBeamformer geometry, zero polar motion and constant
DUT1, explicit EOP coverage, frame-aligned bins. This verifies software phase,
mask/count and precision arithmetic; it is not physical calibration. The
analytic local direction is E=cos(lat)*sin(delta),
N=sin(lat)*cos(lat)*(1-cos(delta)), delta=ERA_target-ERA_sample.
"""

import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_n2_accumulate import chime_tel
from test_n2_per_product_support import (
    BINS,
    FREQ,
    SUBS,
    E,
    F,
    PerProductDump,
    S,
    fixture,
)

from kotekan import runner

# IERS ERA rate (turns per UT1 day); this test needs elapsed time, not an
# absolute ERA implementation or the native telescope phase routine.
SIDEREAL_RATE = 1.00273781191135448
LIGHT_SPEED = 299792458.0
BOOT_NS = 1767287750500000000


def independent_expectation(streams, period, spf, scale):
    rows, cols = np.triu_indices(E)
    tile = (cols // 16) * (cols // 16 + 1) // 2 + rows // 16
    hi, lo = cols // 8, rows // 8
    count_tile = (hi // 8) * (hi // 8 + 1) // 2 + lo // 8
    dish = np.arange(E) % (E // 2)
    east_m = (dish // 16 - 1.5) * 22.0
    north_m = (dish % 16 - 7.5) * 0.3048
    latitude = np.deg2rad(50.0)
    integration_s = S * scale * period * 2.56e-6
    phase_span = 0.0
    result = {}
    for b in range(BINS):
        for f in range(F):
            raw, counts, admitted = [], [], []
            for t in range(SUBS):
                frame, sub = divmod(b * SUBS + t, spf)
                matrix = streams["in_buf"][frame].data[sub, f]
                x = matrix[tile, cols % 16, rows % 16, 0].astype(np.complex128)
                x -= 1j * matrix[tile, cols % 16, rows % 16, 1]
                count_matrix = streams["in_counts_buf"][frame].data[sub, f]
                n = count_matrix[count_tile, hi % 8, lo % 8].astype(np.uint64)
                # Midpoint of the bin minus midpoint of this subintegration.
                dt = (SUBS / 2 - t - 0.5) * integration_s
                delta = 2 * np.pi * SIDEREAL_RATE * dt / 86400.0
                pointing_east = np.cos(latitude) * np.sin(delta)
                pointing_north = (
                    np.sin(latitude) * np.cos(latitude) * (1 - np.cos(delta))
                )
                freq_hz = (800.0 - FREQ[f] * 800.0 / 2048.0) * 1e6
                angle = (
                    -2
                    * np.pi
                    * freq_hz
                    / LIGHT_SPEED
                    * (east_m * pointing_east + north_m * pointing_north)
                )
                # The wire implementation stores single-precision per-input
                # phases; retain that documented rounding before baseline products.
                phase = np.exp(1j * angle).astype(np.complex64).astype(np.complex128)
                baseline_phase = phase[rows] * phase[cols].conjugate()
                phase_span = max(
                    phase_span, float(np.max(np.abs(np.angle(baseline_phase))))
                )
                raw.append(np.where(n > 0, x * baseline_phase, 0))
                counts.append(n)
                admitted.append(
                    bool(streams["in_rfiframemask_buf"][frame].data[sub, f])
                )
            n = np.zeros(len(rows), dtype=np.uint64)
            k = np.zeros(len(rows), dtype=np.uint64)
            total = np.zeros(len(rows), dtype=np.complex128)
            q = np.zeros(len(rows))
            for t in range(0, SUBS, 2):
                if not (admitted[t] and admitted[t + 1]):
                    continue
                a, z = counts[t], counts[t + 1]
                n += a + z
                total += raw[t] + raw[t + 1]
                good = (a > 0) & (z > 0)
                k += good
                diff = raw[t][good] / a[good] - raw[t + 1][good] / z[good]
                q[good] += (
                    a[good].astype(float)
                    * z[good]
                    / (a[good] + z[good])
                    * np.abs(diff) ** 2
                )
            mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
            weight = np.divide(
                n.astype(float) * k, q, out=np.zeros_like(q), where=(q > 0) & (k > 0)
            )
            result[b + 1, int(FREQ[f])] = {
                "mean": mean,
                "weight": weight,
                "count": n * period,
                "k": k,
                "q": q,
            }
    return result, phase_span


@pytest.mark.parametrize("period,spf", [(1, 2), (4, 2), (4, 1)])
def test_nontrivial_rotation_with_heterogeneous_support(tmp_path, period, spf):
    scale = 512
    streams, unrotated = fixture(period, spf, scale)
    expected, phase_span = independent_expectation(streams, period, spf, scale)
    # Guard against a vacuous identity-rotation test.
    assert phase_span > 0.01
    config = {
        "buffer_depth": 3,
        "samples_per_data_set": spf * S * scale,
        "sub_integration_ntime": S * scale,
        "num_local_freq": F,
        "num_elements": E,
        "num_polarizations": 2,
        "num_dishes": E // 2,
        "num_ev": 0,
        "telescope": deepcopy(chime_tel),
        "gps_time": {"frame0_nano": BOOT_NS},
        "eop": {
            "kotekan_update_endpoint": "json",
            "earth_orientation_parameter_table": [
                {
                    "t_inst_ns": BOOT_NS + offset,
                    "delta_UT1_inst": 0.0,
                    "xp_as": 0.0,
                    "yp_as": 0.0,
                }
                for offset in [-3600_000_000_000, 3600_000_000_000]
            ],
        },
    }
    config["telescope"].update(
        frame0_nano=BOOT_NS,
        eop_updatable_config="/eop",
        require_eop=True,
        fatal_eop_out_of_range=True,
        feed_sep_EW=22.0,
        feed_sep_NS=0.3048,
        num_cylinders=4,
    )
    readers = {
        key: runner.ReadChordBuffer(str(tmp_path), frames)
        for key, frames in streams.items()
    }
    for reader in readers.values():
        reader.write()
    output = PerProductDump(
        str(tmp_path), exit_after_n_files=BINS * F, num_elements=E, num_ev=0, num_freq=F
    )
    stage = runner.KotekanStageTester(
        "N2Accumulate",
        {
            "num_freq_per_n2k_frame": F,
            "packet_loss_is_scalar": False,
            "bin_in_ERA": False,
            "num_subintegrations_per_bin": SUBS,
            "variance_mode": "EvenOddPosDef",
            "do_fringestop": True,
            "input_order": "CHIMEBeamformer",
        },
        readers,
        output,
        config,
    )
    stage.run()
    actual = {
        (int(frame.metadata.abs_time_idx), int(frame.metadata.freq_id)): frame
        for frame in output.load()
    }
    assert len(actual) == BINS * F
    receipts = []
    max_rotation_effect = 0.0
    for original in unrotated:
        key = original["bin"], original["freq"]
        ref, got = expected[key], actual[key]
        np.testing.assert_array_equal(ref["count"], original["count"])
        np.testing.assert_array_equal(got.valid_fpga_ticks, original["count"])
        np.testing.assert_allclose(got.vis, ref["mean"], rtol=3e-6, atol=2e-7)
        np.testing.assert_allclose(got.weight, ref["weight"], rtol=2e-5, atol=1e-7)
        assert got.metadata.fpga_start_tick == original["seq"]
        assert got.metadata.frame_length_fpga_ticks == original["length"]
        np.testing.assert_array_equal(got.weight[ref["k"] == 0], 0)
        effect = float(np.max(np.abs(got.vis - original["mean"])))
        max_rotation_effect = max(max_rotation_effect, effect)
        receipts.append(
            {
                "bin": key[0],
                "frequency": key[1],
                "max_mean_error": float(np.max(np.abs(got.vis - ref["mean"]))),
                "max_weight_error": float(np.max(np.abs(got.weight - ref["weight"]))),
                "max_relative_weight_error": float(
                    np.max(
                        np.divide(
                            np.abs(got.weight - ref["weight"]),
                            ref["weight"],
                            out=np.zeros_like(ref["weight"]),
                            where=ref["weight"] > 0,
                        )
                    )
                ),
                "selected_baseline_inputs": [0, 48],
                "selected_count_ticks": int(got.valid_fpga_ticks[48]),
                "selected_usable_pairs": int(ref["k"][48]),
                "selected_expected_mean": [
                    float(ref["mean"][48].real),
                    float(ref["mean"][48].imag),
                ],
                "selected_actual_mean": [
                    float(got.vis[48].real),
                    float(got.vis[48].imag),
                ],
                "selected_expected_weight": float(ref["weight"][48]),
                "selected_actual_weight": float(got.weight[48]),
                "max_mean_change_from_unrotated": effect,
                "count_equal": bool(
                    np.array_equal(got.valid_fpga_ticks, original["count"])
                ),
                "supported_without_precision": int(
                    np.sum((ref["count"] > 0) & (ref["k"] == 0))
                ),
            }
        )
    assert max_rotation_effect > 0.001
    target = os.environ.get("N2_ROTATION_EVIDENCE_DIR")
    if target:
        destination = Path(target)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"period{period}-spf{spf}.json").write_text(
            json.dumps(
                {
                    "scope": "Independent zenith/ENU arithmetic; deterministic zero-polar-motion EOP; no physical calibration",
                    "period": period,
                    "subintegrations_per_frame": spf,
                    "scale": scale,
                    "maximum_baseline_phase_radians": phase_span,
                    "records": receipts,
                },
                indent=2,
            )
            + "\n"
        )
