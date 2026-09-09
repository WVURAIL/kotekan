# N2 time downsampling: count and precision contract

`N2TimeDownsample` accepts consecutive full upper-triangle N2 frames at one
frequency, with legacy scalar support or an explicit `per_product_v1` descriptor. In
scalar mode the valid FPGA-tick count is shared by all products in a frame. It bins the input reference epochs by Earth Rotation Angle (ERA),
fringestops when configured, and combines the supported input means.

For product p and input frame i, let N_i be `n_valid_fpga_ticks`, V_ip the
visibility, w_ip its inverse variance, and a_ip the complex fringe multiplier.
The output is

```
N = sum_i N_i
V_p = sum_i N_i a_ip V_ip / N
w_p = N^2 / sum_i (N_i^2 |a_ip|^2 / w_ip)
```

The precision equation assumes independent errors between input frames. Equal
counts and equal input precisions reduce to `w_out = number_of_frames * w_in`.
With unequal counts the actual squared coefficients must be propagated. A
nominal frame count cannot replace them. Accumulation uses double precision,
with finite representable float output required for supported visibilities.
An unrepresentable, nonpositive or nonfinite output precision becomes zero.

## Versioned per-product support

With `support_mode: per_product_v1` on both buffer descriptors, the appended
`valid_fpga_ticks[p]` array supplies N_ip separately for each product. The same
mean and variance equations apply independently to each product, and the
output support is the exact uint64 sum in descriptor product order. Zero
support ignores only that product's payload; another baseline in the same
frame can remain supported. Scalar valid/PL/RFI fields must be zero and are
unavailable under this schema, while the FPGA interval remains meaningful.
Missing reason splits cannot be reconstructed from the joint valid counts.

Both endpoints explicitly opt into the schema. Ordinary `N2FrameView` callers
refuse it; the raw Python loader requires `support_mode="per_product_v1"` and
checks exact byte size. Legacy frame offsets and default payload bytes remain
unchanged. The descriptor and output writer must preserve the count mode.

There is no common sample count with which to average eigen diagnostics in
heterogeneous mode. The downsampler clears eigenvalues/vectors, sets eigen
method to `none`, residual to -1 and radiometer chi-square to -1. These are
unavailable diagnostics, not measurements of a noise-free matrix.

## Missing support and unavailable precision

- A zero-count input contributes no visibility, variance or eigen diagnostic.
  Its unused payload may contain NaNs or infinities without poisoning later
  supported data. Its elapsed interval and loss counters remain accounted for.
- A positive-count input with zero, negative, NaN or infinite precision retains
  its finite contribution to the mean, but makes the output precision zero for
  that product. Zero precision means unavailable uncertainty; it does not mean
  zero variance or zero signal.
- Supported nonfinite visibility/eigen payload is refused. The stage does not
  silently publish a valid-looking replacement.
- An all-zero-count output has zero visibility and precision, with input flags
  and mask cleared. A later output reusing the same ring slot starts fresh.

## Identity and timing

Both buffers must have matching N2 descriptors. Inputs must have a full upper
triangle with the configured input count, a stable physical/channel frequency,
valid count accounting, nonoverflowing FPGA intervals, matching start-time and
midpoint EOP metadata, and increasing canonical reference EOP epochs. FPGA
intervals and absolute input indices must be consecutive. A gap or reordered
input stops the stage; this implementation does not infer lost intervals.
The loss identity is

```
frame_length = n_valid + n_rfi_only + n_pl
n_rfi_only <= n_rfi <= n_rfi_only + n_pl
```

Dataset and RFI-policy metadata must agree within an output bin. Supported
input flags, masks, gains and eigen methods must also agree. The first
supported frame supplies these fields when earlier frames have no support.

The bin index includes the absolute Earth rotation, preventing frames on
successive days with the same angle from merging. The first observed partial
bin and final unfinished bin are not published. A completed bin older than
`max_age` in accumulated elapsed time is discarded. `bin_eop` describes the
nominal ERA bin center; `time_center_eop` is recomputed from the actual
accumulated FPGA interval midpoint. The local ERA edge fields are also filled.

## Metadata round trip and legacy fixture correction

`N2MetadataFormat` already reserves two uint64 words for `dataset_id`. The raw
reader and writer previously omitted copying those words, and JSON omitted
the field. The corrected reader/writer preserve the existing words without
changing native size, offsets, or legacy frame layout. Old binary dumps written
by the old serializer contain the null ID and remain readable; a lost ID cannot
be reconstructed from those dumps. Missing IDs in old JSON load as null. New
JSON carries the ID explicitly.

The existing `FillIJMissingVisPattern` N2 generator reports two missing ticks:
one received RFI tick and one packet-loss tick. It now fills `n_pl=1`, making
its metadata satisfy the documented count identity. The existing loss and
fringestopping regression remains in place.

## Verification and scientific limits

`tests/test_n2_time_downsample_normalization.py` drives the real CPU stage from
raw N2 frames and compares all products to an independent exact rational
(`fractions.Fraction`) oracle. It covers unequal counts and precisions,
partially/all missing support, invalid precisions, large counts, repeated
buffer reuse, and a one-bin-per-rotation stream crossing multiple rotations.
Refusal tests require both a nonzero process exit and the intended diagnostic.
Nonzero 128-bit dataset IDs exercise the binary metadata round trip.
`tests/test_n2_time_downsample.py` retains the existing missing-tick and
fringestopping checks.

CPU evidence and commands are retained under
`output/downsampler-normalization-2026-09-09` from the workspace root;
reviewable results are in `results/downsampler_normalization_2026-09-09`.
The original binary and original source are preserved before rebuilding.
The initial fixture-setup failure and subsequent engineering runs are kept
separately from the completed verification records.

This validates arithmetic and bookkeeping in the CPU stage. It does not
measure covariance between adjacent frames, receiver calibration, GPU
throughput, telescope integration, or online recovery after gaps. Nonzero
precision is conditional on valid input uncertainty estimates and their
independence. The legacy scalar field cannot encode product-dependent packet loss. The
opt-in per-product interface carries joint valid support, but intentionally
does not infer a per-product packet-loss/RFI reason decomposition. In scalar mode, averaged eigen diagnostics retain the
existing count-weighted convention; they are not a fresh eigendecomposition
of the combined visibility matrix. `radiometer_chi2` is inherited from the
first supported frame rather than re-estimated.
