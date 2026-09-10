"""Check the study's finite-law moments and support oracle independently."""

import importlib.util
from itertools import product
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "study",
    Path(__file__).resolve().parents[1] / "tools/measure_n2_empirical_precision_v1.py",
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_moments_by_finite_enumeration():
    noise = [complex(a, b) for a, b in product(range(-2, 3), repeat=2)]
    for reference in study.product_moments():
        i, j = [list(study.ACTIVE).index(k) for k in reference["inputs"]]
        if i == j:
            z = np.array([abs(study.MU[i] + u) ** 2 for u in noise])
        else:
            z = np.array(
                [
                    (study.MU[i] + u) * (study.MU[j] + v).conjugate()
                    for u, v in product(noise, repeat=2)
                ]
            )
        np.testing.assert_allclose(z.mean(), complex(*reference["mean"]), atol=1e-14)
        np.testing.assert_allclose(
            np.mean(abs(z - z.mean()) ** 2), reference["variance"], atol=1e-14
        )


def test_masked_sums_from_individual_sample_products():
    for case in (0, 1, 2, 3):
        components, present, keep, admit = study.generate(838182, case, 0)
        raw, joint, n, k, _q, mean, weight, use = study.sufficient(
            components, present, keep, admit
        )
        for p, (i, j) in enumerate(zip(study.AR, study.AC)):
            mask = (
                present[..., study.ACTIVE[i] // 8]
                & present[..., study.ACTIVE[j] // 8]
                & keep
            )
            x = components[..., i, 0].astype(np.int64) + 1j * components[..., i, 1]
            y = components[..., j, 0].astype(np.int64) + 1j * components[..., j, 1]
            direct = np.where(mask, x * y.conjugate(), 0)
            np.testing.assert_array_equal(direct.sum(axis=1), raw[:, p])
            np.testing.assert_array_equal(
                mask.sum(axis=1), joint[:, study.ROWS[p] // 8, study.COLS[p] // 8]
            )
            assert n[p] == np.count_nonzero(mask & use[:, None])
            direct_mean = direct[use].sum() / n[p] if n[p] else 0
            np.testing.assert_allclose(mean[p], direct_mean, atol=1e-14)
        if case == 1:
            assert admit.sum() == 7 and use.sum() == 6
        if case == 3:
            probes = study.ROWS == 0
            assert np.all(n[probes] > 0)
            assert np.all(k[probes] == 0)
            assert np.all(weight[probes] == 0)
