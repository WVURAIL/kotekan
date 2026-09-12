# Masked visibility accumulation

`N2Accumulate` normalizes correlator sums by the number of accepted voltage
samples. With `variance_mode: EvenOddPosDef`, it estimates weights from
adjacent even/odd pairs.

For each product, let `N` be the accepted sample count and `k` the number of
accepted pairs with positive counts in both frames:

```text
Q = sum_pairs [n0*n1/(n0+n1) * abs(corr1/n1 - corr0/n0)^2]
V = sum_accepted corr / N
weight = N*k/Q
```

Rejecting either frame with the second-stage mask drops the whole pair. If
only one frame has samples, it contributes to `V` and `N`, but not `Q` or `k`.
The weight is zero if `N`, `k` or `Q` is zero, or if `Q` or the weight is
nonfinite. A valid mean can therefore have zero weight. The output conjugates
the lower-triangular input into upper-triangular order.

`Q/(k*N)` estimates the variance of the mean when sample errors are independent,
have a common variance, and both frames have the same expected normalized
visibility after fringestopping. Masking based on the data can break these
assumptions. Taking its reciprocal does not give an unbiased estimate of inverse
variance.

## Counts and timing

Configured integration lengths and normalization counts use voltage samples;
timing and output counts use FPGA ticks. Input metadata must give a positive
integer number of ticks per sample. The sample period and frequency order must
stay fixed, and all five input streams must have matching time and frequency
metadata. Correlation frames must be consecutive and start on a frame boundary.

Counts must be between zero and `sub_integration_ntime`. The redundant upper
entries in diagonal tiles are ignored. In the default scalar mode, counts must
also be equal across products within each subintegration and frequency. Unequal
counts require `packet_loss_is_scalar: false`, `variance_mode: EvenOddPosDef`,
and an output descriptor with `support_mode: per_product_v1`; see
[per-product support](n2_per_product_support.md).

## Tests

Run the CPU stage tests against the selected build:

```sh
KOTEKAN_BUILD_DIRNAME=build OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -m pytest -q tests/test_n2_accumulate.py \
  tests/test_n2_accumulate_mask_normalization.py tests/test_rfi_masksum.py
```

The tests compare saved output with independently calculated counts, means and
weights. Cases cover masked pairs, zero support, unequal counts between frames,
FPGA tick conversion, frame boundaries and invalid input.
