# Empirical CPU accumulation precision

`tools/measure_n2_empirical_precision_v1.py` freezes and runs a bounded
independent synthetic experiment through the actual CPU `N2Accumulate` stage.
`tools/audit_n2_empirical_precision_v1.py` reconstructs every archived native
input and output without importing the experiment's generator or oracle.

The completed release is
`../results/pathfinder_empirical_precision_2026-09-09`. Its plan, native I/O,
source and executable snapshots, full receipts, audit, and summary figure are
retained. It does not modify the accumulator implementation.

## Model and estimand

For active input `i`, each sample is `x_i = mu_i + u_i`, where the independent
real and imaginary components of `u_i` are uniform on the integers
`{-2,-1,0,1,2}`. Samples and active inputs are independent. Four active inputs
have means `0, 0, 1+i, -i`; the other 60 inputs contain exactly zero voltage.
All mask mechanisms are independent of voltage values. Packet support is
shared by eight-input groups, and a product uses the actual intersection of
both input masks and the common fine mask. Frame rejection removes both
members of an even/odd pair.

Write `z=x_i conjugate(x_j)`. For a cross product, its mean is
`mu_i conjugate(mu_j)` and its complex variance `E|z-Ez|^2` is
`16 + 4|mu_i|^2 + 4|mu_j|^2`. For an auto product, `z` is real, its mean is
`4+|mu_i|^2`, and its real variance is `5.6+8|mu_i|^2`. A separate finite
enumeration of all 25 auto or 625 cross noise combinations verifies these
moments. Complex variance is the sum of real and imaginary variances, not
either component variance alone.

Conditional on the masks, the count-weighted output is the mean of `N`
independent identically distributed sample products. Its variance is therefore
`sigma_z^2/N`. For each admitted pair with counts `n0,n1>0`, define

```
q_pair = n0*n1/(n0+n1) * |mean0-mean1|^2
Q      = sum(q_pair)
K      = number of such supported pairs
weight = N*K/Q
```

Equal within-pair expected visibility and independent sample errors imply
`E[q_pair | masks] = sigma_z^2`, so `Q/(N*K)` is an unbiased estimate of
`sigma_z^2/N` when `K>0`, including realizations with `Q=0`. For `Q>0`
this equals the reciprocal reported weight; at `Q=0` the implementation emits
zero precision, so this identity is unavailable and the primary comparison
fails rather than conditioning on or excluding that realization. No primary
realization in this release has zero precision.
One-sided admitted samples still contribute to the
mean and `N`, but do not create a supported pair in `K`. The experiment tests
the reciprocal reported weight, **not** the mean weight: in general
`E[1/Vhat] != 1/E[Vhat]`. It does not assume calibrated confidence coverage
or `E[weight*|error|^2]=1`.

## Frozen experiment

Each bin is one 16,384-sample upgrade frame, with eight 2,048-sample
subintegrations, at one frequency in the native 64-input count/correlation
layout. There are 1,024 fresh bins for each of three masks: full support,
independent heterogeneous packet/fine support with exactly one randomly
selected frame rejection, and deterministic masks containing both one-sided
and fully supported pairs. Four products are primary: `(0,0)`, `(0,1)`,
`(0,8)`, and `(8,16)`. Fringestopping and ERA binning are disabled.

For each of these 12 rows, two aggregate ratios use the sum of the exact
conditional variances as denominator: the sum of squared errors about the
known mean, and the sum of reciprocal output weights. The fixed acceptable
range `[0.85,1.15]` is an engineering tolerance, not a 95% or simultaneous
confidence bound. All realizations are retained. Unexpected missing or
nonfinite precision fails the primary reciprocal comparison; it is never
silently excluded. A separate eight-bin guard requires positive-support means
with no supported pairs to report zero precision.

The release passes all 24 predeclared aggregate comparisons. Squared-error
ratios span **0.9540 to 1.0923**, and reciprocal-weight ratios span
**0.9438 to 1.0140**. The two observed estimates are compared separately with
the known model expectation; their mutual ratio is not constrained to 15%.
For example the mixed-mask auto row gives 1.0923 and 0.9438, respectively.
An individual bin's variance estimate can be much noisier than these ensemble
averages; two to four supported pairs do not make a precise per-bin variance
measurement.

All 19,228,440 runtime arithmetic comparisons pass. The independent recount
checks 3,080 archived native output frames and passes 9,067,611 comparisons.
This includes all 2,080 per-product counts per frame, active visibility sums,
algebraically independent pair differences, seed-regenerated integer sample
hashes, native HDF5 inputs, and the summary's primary aggregate ratios.
The 24 guard probe checks pass. No primary precision is zero or nonfinite.

## Reproduction and limits

The release saves `evaluation-command.json` and a Docker image identity. It
uses the existing CPU binary at
`build/pathfinder-normalization-cpu/kotekan/kotekan`, a one-CPU quota, and one
thread for BLAS/OpenMP. There is no GPU or network access. The mounted source
is read-only. In that runtime, with a fresh output path:

```
python3 tools/measure_n2_empirical_precision_v1.py freeze OUTPUT
python3 tools/measure_n2_empirical_precision_v1.py run OUTPUT
python3 tools/audit_n2_empirical_precision_v1.py OUTPUT
```

`freeze` binds seeds, sample counts, moments, masks, acceptance, NumPy version,
sources and executable hashes before evaluation. `run` refuses changed
sources or an existing batch directory. `smoke` uses separate engineering
seeds and four-bin counts; it is not evaluation evidence. The design's unit
checks live in `tests/test_n2_empirical_precision_design.py`.

These results check accumulation under the stated finite stationary law.
They do not establish production mask independence, temporal covariance,
physical input mapping, analog transfer, real sky/noise variance, physical
coverage, full GPU processing, `N2TimeDownsample` ensemble propagation,
fringestopping, ERA-bin behavior, hardware deadlines or live Pathfinder
acceptance. The frozen fine-detector campaign is separate.
