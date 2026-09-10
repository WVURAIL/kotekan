# Composed CPU visibility precision

`tools/measure_n2_composed_precision_v1.py` generates fresh independent integer
voltages and masks, runs the actual CPU `N2Accumulate`, and serializes its
native outputs unchanged into the actual CPU `N2TimeDownsample`. This checks
the composed numerical operation through native I/O. It does not run a jointly
scheduled live pipeline, acquire telescope voltages, or measure physical
variance or operational deadlines.

## Frozen model and grouping

Four inputs in a native 64-input layout have complex voltage
`x_i = mu_i + u_i`, with means `0,0,1+i,-i`. The real and imaginary components
of each `u_i` are independent uniform integers on `{-2,-1,0,1,2}`. All sample
times, active inputs and source frames have independent draws. Other inputs
contain exactly zero voltage. Actual product support is the intersection of
the two packet-group masks and the common fine mask; an upstream frame gate
rejects both members of an even/odd pair. All masks are independent of voltage
values. No source or output draw is selected by its value.

Each source frame contains **16,384 samples**, split into eight subintegrations
of 2,048 samples. The accumulator produces one per-product count/mean/precision
frame per source frame. The downsampler uses 500,000 bins per Earth rotation;
both stages have fringestopping disabled and the accumulator uses fixed bins.
One frozen time segment has 70 source frames. Indices 0--2 belong to its first
observed ERA group and are skipped for startup alignment. Indices 3--68 form
16 disjoint complete output groups: fourteen contain four source frames and
two contain five. Source index 69 triggers emission of the final complete
group and is not itself emitted. Exact source start ticks, group membership
and relative ERA identities are frozen before any evaluation voltages.

The two primary mask cases each use 64 independently generated time segments,
giving 1,024 complete output groups per case. Reusing the declared time lattice
does not reuse voltage samples or generator identities. Output groups within
a segment also use disjoint source frames. Product rows within a group share
inputs; no assumption of independence across product rows is made.

The full-support case admits every sample. The heterogeneous case combines
independent packet/fine masks, one-sided even/odd support, and zero-count source
fragments. The first member of alternating groups either has zero counts for
all products or loses packet group zero while other groups remain supported.
This tests zero-count handling and heterogeneous product counts in the actual
downsampler. A separate 16-group guard has one positive-count upstream frame
with zero precision for products involving input group zero. Its supported
control product `(8,16)` retains ordinary finite precision.

## Variance estimand

For `z=x_i conjugate(x_j)`, a cross product has mean
`mu_i conjugate(mu_j)` and complex variance
`sigma_z^2 = E|z-Ez|^2 = 16+4|mu_i|^2+4|mu_j|^2`. An auto product is real,
with mean `4+|mu_i|^2` and variance `5.6+8|mu_i|^2`. Complex variance includes
both real and imaginary components. Independent finite enumeration verifies
these moments in the accumulator study.

Within an output group let upstream product counts be `N_i`, means `V_i` and
precisions `w_i`, and let `N=sum(N_i)`. The composed mean and propagated
reciprocal precision are

```
V_out    = sum(N_i V_i) / N
1/w_out  = sum(N_i^2 / w_i) / N^2
```

The latter is available only when every positive-count source frame has finite
positive precision. Zero-count source frames contribute neither a mean nor a
variance term. Under the declared independent stationary voltage/mask law, the
ideal unrounded mean is the mean of `N` individual products, with conditional
variance `sigma_z^2/N`.

For each supported upstream even/odd pair with sample counts `n0,n1`, define
`q=n0*n1/(n0+n1)*|mean0-mean1|^2`. Let `Q_i` be their sum and `K_i` the number
of supported pairs. Equal within-pair expected visibility and independent
samples give `E[Q_i | masks]=K_i*sigma_z^2`. Consequently

```
E[ sum(N_i^2 * Q_i/(N_i*K_i)) / N^2 | masks ] = sigma_z^2/N
```

when every positive-count source frame has `K_i>0`. This ideal identity includes
`Q_i=0`. For positive `Q_i`, `Q_i/(N_i*K_i)` equals reciprocal upstream precision
before float32 rounding. At zero `Q_i` the implementation emits zero precision;
its reciprocal is unavailable. Such a primary event fails qualification rather
than being excluded or conditioned away. The guard separately tests intentional
unavailable precision while requiring the supported mean and count to survive.

The two ensemble comparisons use the same denominator, the sum of ideal
`sigma_z^2/N` across all predetermined complete output groups. Their numerators
are the sum of squared final-mean errors about the known population mean and
the sum of reciprocal final weights. Each ratio must lie in `[0.85,1.15]`.
This is a fixed engineering tolerance, not a confidence interval or confidence
coverage test. Mean/variance-estimator independence is unnecessary for these
separate comparisons; unbiasedness of reciprocal precision does not imply
unbiased precision. Individual variance estimates remain noisy.

Float32 rounding is a separate numerical issue. The study checks all ten
active-product source and final means/weights against arithmetic references,
all 2,080 counts at both stages exactly, native replay bytes, and all time/group
identities. Fixed relative/absolute arithmetic tolerances are recorded in the
plan. The ideal statistical variance formula is not an assertion of exact
unbiasedness after float32 rounding.

## Retained evaluation

The frozen evaluation contains 8,960 primary source frames and 70 guard source
frames. Their native outputs produce 2,048 primary downstream groups and
16 guard groups. All **16 predeclared ensemble comparisons** pass the fixed
engineering tolerance. The actual stage outputs pass **23,351,580 arithmetic
checks**, with no failed batch. An independently authored native-data audit
passes **77,297,995 checks**. It regenerates every voltage/mask stream without
importing the study generator, reconstructs joint counts and moments, checks
exact rational ERA membership and count-weighted propagation, authenticates
the replay boundary, and reconstructs all eight result rows. The guard passes all 48 unavailable-product
checks and all 16 supported-control checks. No primary probe group contains
an unavailable positive-count upstream precision; no such realization is
excluded. The table uses all 1,024 predetermined groups in each case.

Both ratios below divide by the sum of ideal conditional variances
`sigma_z^2/N`: the squared-error numerator uses the known population mean,
while the reciprocal-precision numerator uses the actual reported final
weights. The displayed count ranges are actual per-product joint support
across output groups, after masking.

| Mask case | Product | Samples per output group | Squared-error ratio | Reciprocal-precision ratio |
|---|---|---:|---:|---:|
| Full | `(0, 0)` | 65,536--81,920 | 1.055173 | 1.010798 |
| Full | `(0, 1)` | 65,536--81,920 | 1.039448 | 1.000039 |
| Full | `(0, 8)` | 65,536--81,920 | 0.968929 | 0.996191 |
| Full | `(8, 16)` | 65,536--81,920 | 1.038299 | 1.007824 |
| Heterogeneous | `(0, 0)` | 30,950--41,512 | 1.019094 | 1.001705 |
| Heterogeneous | `(0, 1)` | 30,950--41,512 | 1.000521 | 0.999761 |
| Heterogeneous | `(0, 8)` | 22,120--29,955 | 1.007340 | 0.998795 |
| Heterogeneous | `(8, 16)` | 18,223--27,487 | 0.994555 | 0.994339 |

Measured squared-error ratios span 0.968929--1.055173; reciprocal-precision
ratios span 0.994339--1.010798. These are empirical checks of the declared
synthetic model and composed numerical operation, not calibrated confidence
bounds or physical variance acceptance. The canonical figure is
`figures/n2-composed-precision.pdf` within the evidence release.

## Reproduction and evidence

The output release is
`../results/pathfinder_composed_precision_2026-09-09`. It records the frozen
plan, sources and CPU executable, Docker runtime identity, commands, complete
native HDF5/accumulator/downsampler I/O, replay hashes, source sufficient
statistics, per-output receipts and a summary. The exact native accumulator
bytes are preserved at the downstream replay boundary. The runtime uses one
CPU with no network or GPU access. Existing implementations, binaries and
earlier releases are not modified.

```
python3 tools/measure_n2_composed_precision_v1.py freeze NEW_OUTPUT
python3 tools/measure_n2_composed_precision_v1.py run NEW_OUTPUT
```

Use the recorded container command with its source mount, numerical runtime,
CPU limits and `KOTEKAN_BUILD_DIRNAME` to reproduce the native stages. The
driver explicitly refuses changed plan hashes, sources, numerical runtime,
time/group membership, source frame size or primary output counts, including
under optimized Python. The `smoke` command uses separate seeds and one batch
per case. It tests execution and arithmetic; its small-sample ensemble ratios
are not primary evaluation. All primary failures are retained without retries
or seed/count/tolerance changes.

`tests/test_n2_composed_precision_design.py` checks the composed conditional
variance identity by exact finite enumeration, verifies fixed membership and
mask guards, and exercises six provenance/runtime/geometry refusal paths.
`tools/plot_n2_composed_precision_v1.py` renders the summary without generating
or selecting new trials. It shows point estimates and the fixed engineering
tolerance, without uncertainty whiskers. The summary's descriptive `mc_se`
fields use model-centered residuals (`observed-target`); they are not generic
ratio standard errors or confidence bounds.

Physical sky/noise covariance, value-dependent production masks, calibrated
fringestopping, arbitrary ERA boundary placement, USB acquisition, real input
health, shared GPU load, sustained operation and live Pathfinder acceptance
remain separate requirements.
