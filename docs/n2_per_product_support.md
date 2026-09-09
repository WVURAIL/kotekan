# Per-product visibility support

The optional `N2Accumulate` mode `packet_loss_is_scalar: false` requires an
output N2 descriptor with `support_mode: per_product_v1`. The scalar default
and its byte layout remain unchanged. The new mode requires `EvenOddPosDef`
without debug accumulation and uses a count for each product. No production
configuration is enabled by this change.

For each product, accumulate admitted correlator sums and joint valid counts:
`V = sum(C_t) / N`. Each admitted even/odd pair with two positive counts adds
`n0*n1/(n0+n1) * abs(C1/n1 - C0/n0)^2` to Q and one observation to k.
The precision is `N*k/Q`, when all required quantities exist and are finite.
One-sided support contributes to the mean but not k/Q; a zero-count raw payload
is unused, even when its stored integers are nonzero. Rejection of either
member by the second-stage frame mask rejects the whole pair. The new sums
and variance arithmetic use double precision, including optional fringestop
rotation. Marginal packet-loss counts never substitute for joint support.

These equations inherit the independent-error, common per-sample variance and
equal within-pair expected visibility assumptions in `n2_mask_normalization.md`.
Data-dependent masks can violate them. This is arithmetic verification, not
physical variance or retained-science calibration.

## Native mapping

`cudaCorrelator.cpp` passes contiguous `[T,F,P,D]` voltage memory directly to
`n2k::Correlator`. Its station index flattens `[P,D]`. The PL producer passes
`[T/64,F,P,D/8,Tlo64]` bits in the same order, with one bit per eight adjacent
dish positions within one polarization. `n2k/Correlator.hpp` documents its
row-major 16x16 lower-triangular tiles; `n2k/pl_kernels.hpp` and
`src_lib/pl_1bit_correlator.cu` describe and write row-major 8x8 count tiles.

For a native lower-triangular product `(i,j)`, set `g_i=i//8`, `g_j=j//8`,
`b_i=g_i//8`, `b_j=g_j//8`. Its count offset is
`64*(b_i*(b_i+1)//2+b_j) + 8*(g_i%8) + g_j%8`.
The correlation offset uses the analogous block size 16. Only true science
entries are used; redundant upper entries within diagonal tiles are ignored.
The N2 upper-triangular value is the conjugate at product `(j,i)`.
This is native source-code ordering. Physical input-map certification and
actual GPU-to-output replay remain separate acceptance requirements.

## Counts and unavailable reasons

The appended uint64 `valid_fpga_ticks[num_products]` follows exact descriptor
product order. Arithmetic counts voltage samples; export alone multiplies by
the declared integer ticks per sample. Counts cannot exceed the frame interval.

Legacy scalar valid, packet-loss, RFI and RFI-only counters are unavailable in
this mode and held at zero sentinels. Those zeros are not measurements.
The descriptor is explicit: the updated `N2FrameView` and Python loader
require opt-in for this extension. Scalar validity/loss-fraction metrics are not emitted. Scalar
PL/RFI inputs remain synchronized and range checked, but are not promoted to
per-product reasons. RFI configuration metadata still records actual settings.

Per-product loss attribution needs the sample-by-sample intersection of raw
packet bits for both groups before masks are applied. The group counts in
`DtvInputHealth` cannot reconstruct this from equal marginal totals.

## CPU verification

`tests/test_n2_per_product_support.py` builds complex voltage samples and
Boolean masks before directly calculating dense products and joint counts,
independently of the blocked layouts. Its 128 inputs cross count/correlation tile
boundaries. Fixtures cover disjoint groups with equal marginal counts, wholly
missing inputs and bins, one-sided pairs, RFI/loss overlap, second-stage
rejection, unused payloads, two frequencies, ring reuse and pairs crossing
input frames. One/four FPGA-tick periods and logically repeated counts up to
65536 exercise unit conversion and widened arithmetic. Repetition is an
arithmetic fixture, not new independent data. Exact counts and numerical
means/precisions are compared with independent calculations.
Three additional CPU cases in `tests/test_n2_per_product_rotation.py` enable
nontrivial fringestopping with an explicit deterministic EOP table. An
independent zenith/ENU direction calculation checks per-product counts, means
and precision for periods one/four and pairs spanning input frames. The
largest baseline phase is 0.109816 rad; maximum mean absolute error is 2.384e-7
and maximum precision relative error is 5.912e-8. This checks software rotation
in fixed-bin, zero-polar-motion geometry, not measured telescope phases or
ERA-bin-boundary behavior. The explicit EOP table matters: absent EOP data can
otherwise make a superficially enabled rotation test an identity operation.

## Storage and downstream consumers

`N2TimeDownsample` applies the count-weighted mean and propagated independent-
frame precision formula separately for each product. Supported unknown input
precision makes output precision unavailable while retaining its finite mean.
All-zero support ignores its payload. Eigen diagnostics are unavailable in the
new mode. Product identity, dataset, time and support-mode mismatches are refused.

`hdf5N2Write` must explicitly match the descriptor support mode. The scalar
default remains `CHORD_0.0`. The opt-in `CHORD_0.1` output includes
`valid_fpga_count_per_product(frequency,product,time)` in uint64 FPGA ticks and
omits the eight unavailable legacy scalar count/fraction datasets. It records
scalar-support and loss-reason availability explicitly. Same-size reordered
product subsets are not interchangeable.

Native raw files retain their existing layout: each frame has a uint32
metadata-size header, serialized metadata, and its configured payload. The
reader now checks exact record framing and each header, including multi-frame
files produced by `rawFileWrite`. The format does **not** embed the descriptor,
support mode or product order. Exact-length checks catch malformed or many
wrong-mode inputs, but are not universal format authentication. Bind raw files
to the exact saved configuration; use the versioned HDF5 output for explicit
support-mode interchange. Do not use an old reader that can silently drop
extended bytes.

The GPU count producer's supported geometries and removal of its full-count
fallback are documented in `n2-count-geometry.md`. The compile and CPU mapping
checks do not establish kernel runtime behavior or live throughput.
