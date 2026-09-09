# Accumulated visibility normalization after masking

`N2Accumulate` normalizes admitted raw correlation sums by their surviving
voltage-sample count. Its `EvenOddPosDef` variance estimator now counts only
admitted even/odd pairs with positive support in both members. A missing
variance observation must not increase inverse-variance weight.

For each product, let `N` be the total admitted valid voltage-sample count,
`k` the number of pairs with both counts positive, and

```
Q = sum_pairs n0*n1/(n0+n1) * |corr1/n1 - corr0/n0|^2
V = sum_admitted corr / N
weight = N*k/Q
```

A pair rejected by either second-stage frame mask contributes neither to the
mean nor the variance. A one-sided positive-count pair still contributes its
supported samples to `V` and `N`, but contributes neither `Q` nor `k`.
Zero support, zero usable pairs, zero `Q`, or nonfinite precision produces
weight zero. A supported mean can therefore have unavailable precision; zero
weight does not mean that its noise variance has been measured to be zero.
Output layout retains the established lower-to-upper-triangle conjugation.

The former implementation used all nominal pairs in the numerator, including
pairs absent from `Q`. Deterministic masked cases reproduce weights two to
four times the independent expected value. The multiplication `n0*n1` also
occurred in int32 before conversion; it now widens before arithmetic, preventing
count-product overflow in the 65,536-sample regression.

The statistical interpretation assumes independent sample errors, a common
per-sample variance over the accumulation, and equal expected normalized means
within each differenced pair after any fringestopping. Under those assumptions,
`Q/k` estimates the common variance and `Q/(k*N)` estimates variance of the
integrated mean. The inverse of that estimate is not unbiased precision.
Data-dependent masking may violate these assumptions. Correct arithmetic does
not establish calibrated thermal covariance or science-transfer response.

## Time and support contracts

Configured `samples_per_data_set` and `sub_integration_ntime` count voltage
samples. The incoming correlation metadata supplies the number of FPGA ticks
per subintegration. Dividing by `sub_integration_ntime` must give a positive
integer tick period. This period is frozen on first input, together with the
coarse-frequency ordering.

Visibility/count/variance arithmetic remains in voltage samples. Frame
continuity, accumulation bins, EOP timing, fringe times and output timestamp
spans use FPGA ticks. Exported `n_valid_fpga_ticks`, packet-loss and RFI counts
are multiplied by the tick period, while the visibility denominator is not.
The checks cover periods of one and four ticks per voltage sample.

All five correlation/count/mask streams must supply matching frequency and
time metadata. Correlation frames must be globally aligned to their configured
frame span and consecutive; missing, duplicate, shifted or reordered input
must not reuse a saved even frame. Tick conversions are checked for overflow.

Only scalar support across inputs is implemented. Every actual lower-triangle
entry in the count matrix must agree and lie in
`[0, sub_integration_ntime]`; redundant upper entries of diagonal tiles are
ignored. Diagnostic loss counts must be in range and output totals consistent.
Different per-baseline supports are refused, not normalized using the first
baseline's count. Implementing that case requires per-baseline counts,
normalization, variance and output metadata together.

## Reproduce the CPU regression

Use a CPU Kotekan build with tests enabled. The Python runner reads
`KOTEKAN_BUILD_DIRNAME`; it must identify the build whose binary is tested.
The new tests use the actual `N2Accumulate` stage, metadata-bearing input
buffers and serialized N2 output. No CUDA detector or live calibration is used.

```sh
KOTEKAN_BUILD_DIRNAME=build/pathfinder-normalization-cpu \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -m pytest -q tests/test_n2_accumulate.py \
  tests/test_n2_accumulate_mask_normalization.py tests/test_rfi_masksum.py
```

The deterministic test fixes complex baseline phases, rotates eight cases
across three frequencies and eight accumulation bins, and checks the entire
output matrix. Cases include clean support, frame rejection, one-sided and
both-sided zero counts, no complete variance pair, zero differences, unequal
counts, and mixed first-stage RFI/packet loss with frame rejection. Separate
malformed-input tests check the support/time identity refusals. An optional
`N2_NORMALIZATION_EVIDENCE_DIR` saves compact measured output receipts for
review; ordinary tests do not require it.

The corrected CPU build passes 90 tests: 45 existing accumulation/mask-count
tests and 45 new normalization cases. The latter include 32 successful case
combinations, 12 malformed-input refusals and one redundant-upper-count
acceptance case. All 96 emitted regression receipts are retained, including
24 with even/odd pairs crossing input frames.

The separate evidence release in the parent workspace,
`results/pathfinder_normalization_2026-09-09`, retains the original-binary
failures, corrected-binary results, exact source/test snapshots, runtime and
build identities, and a before/after figure. This is local implementation
validation. GPU-to-accumulator replay, full shared-GPU timing, telescope
calibration, reason-coded input health and shadow operation remain pending.

`N2TimeDownsample` is a separate downstream stage with an unresolved
zero-weight/nominal-frame normalization issue. The direct fengine output
configuration does not invoke it; do not assume this change validates paths
that add that stage. Existing int32 visibility accumulation and per-pair
fringestop rounding are also outside this change's arithmetic scope.
