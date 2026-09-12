# Masked visibility accumulation

`N2Accumulate` produces normalized visibilities and estimated inverse-variance
weights from admitted correlator sums and valid voltage-sample counts.

For each product, let `N` be the admitted sample count, `k` the number of
admitted even/odd pairs with positive support in both members, and

```text
Q = sum_pairs n0*n1/(n0+n1) * abs(corr1/n1 - corr0/n0)^2
V = sum_admitted corr / N
weight = N*k/Q
```

A second-stage rejection of either member excludes the whole pair. One-sided
support contributes to `V` and `N`, but not `Q` or `k`. Zero support, no usable
pairs, zero `Q`, or nonfinite precision produces zero weight. A supported mean
can therefore have unavailable precision. Output products retain the
lower-to-upper-triangle conjugation.

The variance estimate assumes independent sample errors, a common per-sample
variance, and equal expected normalized means within each differenced pair
after fringestopping. Under these assumptions, `Q/(k*N)` estimates variance of
the accumulated mean. Its reciprocal is not an unbiased precision estimate;
data-dependent masking can violate the assumptions.

## Counts and timing

Configured integration lengths and arithmetic counts use voltage samples.
Input metadata defines a positive integer number of FPGA ticks per sample.
Timing and exported counts use FPGA ticks. The tick period and coarse-frequency
order must remain fixed; all five input streams must have matching coordinates.
Correlation frames must be consecutive and aligned to their configured span.

The default scalar-support mode requires equal, in-range counts in every true
lower-triangular entry; redundant upper entries in diagonal tiles are ignored.
Heterogeneous support requires `packet_loss_is_scalar: false` and an output
descriptor with `support_mode: per_product_v1`; see
[per-product support](n2_per_product_support.md).

## Tests

Run the CPU stage regressions against the selected build:

```sh
KOTEKAN_BUILD_DIRNAME=build OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -m pytest -q tests/test_n2_accumulate.py \
  tests/test_n2_accumulate_mask_normalization.py tests/test_rfi_masksum.py
```

The tests compare serialized stage output with independent count, mean and
precision calculations, including masked pairs, zero support, unequal counts,
FPGA tick conversion, frame boundaries and malformed-input refusals.
