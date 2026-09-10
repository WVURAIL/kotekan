"""Exact finite-law composition and time/mask identities for the CPU study."""

import json
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import measure_n2_composed_precision_v1 as study


def test_exact_composed_mean_and_variance_estimator_identity():
    # Three independent source means have counts2,3,4 and one unequal-sized
    # even/odd pair apiece. The finite population includes Q=0 realizations;
    # never condition them out when checking the ideal unbiased estimator.
    observed_mse = observed_variance = Fraction(0)
    counts = (2, 3, 4)
    total = sum(counts)
    for draw in product((-1, 2), repeat=total):
        offset = 0
        weighted_mean = variance_numerator = Fraction(0)
        for count in counts:
            values = draw[offset : offset + count]
            offset += count
            mean = Fraction(sum(values), count)
            delta = values[0] - Fraction(sum(values[1:]), count - 1)
            q = Fraction(count - 1, count) * delta**2
            weighted_mean += count * mean
            variance_numerator += count * q
        observed_mse += (weighted_mean / total - Fraction(1, 2)) ** 2
        observed_variance += variance_numerator / total**2
    target = Fraction(9, 4) / total
    assert observed_mse / (2**total) == target
    assert observed_variance / (2**total) == target


def test_frozen_group_membership_and_mask_boundary_guards():
    frozen = study.timeline()
    assert frozen["source_frames_per_batch"] == 70
    groups = frozen["complete_groups"]
    assert len(groups) == 16
    complete = [i for g in groups for i in g["source_indices"]]
    assert complete == list(range(3, 69))
    assert frozen["dropped_initial_source_indices"] == [0, 1, 2]
    assert frozen["final_flush_trigger_source_index"] == 69
    assert sorted(len(g["source_indices"]) for g in groups) == [4] * 14 + [5] * 2
    assert all(
        frozen["source_bin_keys"][i] == g["relative_era_bin"]
        for g in groups
        for i in g["source_indices"]
    )
    plan = {"timeline": frozen, "seed": 202609091600}
    # Even-group member0 is wholly unsupported; odd-group member0 loses only
    # packet group0. The unavailable-precision guard has positive sample count
    # but no supported even/odd pair for products involving group0.
    _, packet, _, _, _, _ = study.source_recipe(
        plan, 1, 0, groups[0]["source_indices"][0]
    )
    assert not packet.any()
    _, packet, _, _, _, _ = study.source_recipe(
        plan, 1, 0, groups[1]["source_indices"][0]
    )
    assert not packet[:, :, 0].any() and packet[:, :, 1:].any()
    component, packet, fine, admit, _, _ = study.source_recipe(
        plan, 2, 0, groups[0]["source_indices"][1]
    )
    _raw, _joint, count, pairs, _q, _mean, weight, _use = study.base.sufficient(
        component, packet, fine, admit
    )
    unavailable = study.base.ROWS == 0
    assert np.all(count[unavailable] > 0)
    assert np.all(pairs[unavailable] == 0)
    assert np.all(weight[unavailable] == 0)
    control = (study.base.ROWS == 8) & (study.base.COLS == 16)
    assert np.all(count[control] > 0) and np.all(pairs[control] == 4)


@pytest.mark.parametrize(
    "fault", ["hash", "source", "runtime", "count", "timeline", "frame"]
)
def test_frozen_plan_refuses_identity_changes(tmp_path, monkeypatch, fault):
    plan = {
        "source_sha256": {"sentinel": "original"},
        "numpy_version": np.__version__,
        "batches_per_primary_case": 64,
        "primary_output_groups_per_case": 1024,
        "timeline": {"sentinel": "time-only"},
        "geometry": {"source_frame_samples": 16384},
    }
    monkeypatch.setattr(
        study,
        "hashes",
        lambda: {"sentinel": "changed" if fault == "source" else "original"},
    )
    monkeypatch.setattr(study, "timeline", lambda: {"sentinel": "time-only"})
    if fault == "runtime":
        plan["numpy_version"] = "different-runtime"
    if fault == "count":
        plan["primary_output_groups_per_case"] = 1023
    if fault == "timeline":
        plan["timeline"] = {"sentinel": "changed"}
    if fault == "frame":
        plan["geometry"]["source_frame_samples"] = 12288
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    digest = "invalid" if fault == "hash" else study.base.sha(tmp_path / "plan.json")
    (tmp_path / "plan.sha256").write_text(digest + "\n")
    with pytest.raises(ValueError):
        study.validate_plan(tmp_path)
