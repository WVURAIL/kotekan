# Per-product sample counts

When input masks differ, each visibility product can have a different number
of valid samples. To use separate counts in `N2Accumulate`, set
`packet_loss_is_scalar: false`, `variance_mode: EvenOddPosDef`, and
`support_mode: per_product_v1` on the output N2 descriptor. Debug accumulation
is not supported in this mode. Scalar counts remain the default.

Any supplied dataset IDs must agree. The RFI frame mask must include its
enabled flag and finite thresholds. The dataset and RFI settings must stay
fixed until the stage restarts.

Each product uses the samples accepted for both inputs to calculate its mean
and even/odd variance estimate; see [masked accumulation](n2_mask_normalization.md).
Separate packet-loss totals for the two inputs cannot give this joint count.
Correlator sums with zero counts are ignored.

## Product order and counts

In native `[polarization, dish]` order, packet-loss groups contain eight
adjacent dishes within one polarization. For lower-triangular product `(i,j)`,
define `g_i=i//8`, `g_j=j//8`, `b_i=g_i//8`, and `b_j=g_j//8`. Its native count
offset is `64*(b_i*(b_i+1)//2+b_j) + 8*(g_i%8) + g_j%8`. The correlation offset
uses block size 16. Only lower-triangular entries are used. Without reordering,
output upper-triangular product `(j,i)` stores the conjugate. Configured output
reordering permutes the counts and conjugates visibilities as needed.

The frame appends uint64 `valid_fpga_ticks[num_products]` in descriptor product
order. Counts are converted from voltage samples to FPGA ticks at output and
cannot exceed the frame interval. Scalar valid, packet-loss, RFI and RFI-only
counters are set to zero to indicate that they are unavailable. Scalar
loss-fraction metrics are omitted. Incoming scalar diagnostic counts are still
checked for synchronization and range.

## Storage and consumers

`N2FrameView` and the Python loader must opt in to reading per-product counts.
Readers that do not support this mode reject it.

`hdf5N2Write` requires a matching `support_mode`. It writes `CHORD_0.1` files with
uint64 `valid_fpga_count_per_product(frequency,product,time)` in FPGA ticks and
attributes marking scalar counts and loss reasons as unavailable. Scalar
count and fraction datasets are omitted. Scalar output remains `CHORD_0.0`.
Product order and support mode must stay fixed within a file.

Raw files contain a uint32 metadata-size header, serialized metadata and payload
for each frame. They do not include the descriptor, support mode or product
order, so reading them requires the configuration that wrote them. HDF5 files
store this information with the data.

## Tests

`tests/test_n2_per_product_support.py` compares stage output with products and
joint counts computed directly from voltage and mask fixtures. Cases
cover tile boundaries, disjoint packet groups, missing inputs, one-sided pairs,
frame rejection, ring reuse and FPGA tick conversion.
`tests/test_n2_per_product_rotation.py` checks fringestopping against an
independent phase calculation. Descriptor and HDF5 tests check layout, product
order and stored counts.
