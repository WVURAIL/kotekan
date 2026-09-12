# Per-product visibility support

`N2Accumulate` supports heterogeneous joint counts with
`packet_loss_is_scalar: false`, `variance_mode: EvenOddPosDef`, and an output
N2 descriptor with `support_mode: per_product_v1`. Debug accumulation is not
supported in this mode. The scalar default and its byte layout are unchanged.

Each product uses its own admitted count, mean and even/odd variance estimate;
see [masked accumulation](n2_mask_normalization.md). Marginal packet-loss counts
cannot substitute for joint support. Zero-count correlator payload is unused.

## Product order and counts

Native voltages flatten `[polarization, dish]`. Packet-loss groups contain eight
adjacent dishes within one polarization. For lower-triangular product `(i,j)`,
define `g_i=i//8`, `g_j=j//8`, `b_i=g_i//8`, and `b_j=g_j//8`. Its native count
offset is `64*(b_i*(b_i+1)//2+b_j) + 8*(g_i%8) + g_j%8`. The correlation offset
uses the analogous block size 16. Only true lower-triangle entries are used;
output upper-triangular product `(j,i)` stores the conjugate.

The appended uint64 `valid_fpga_ticks[num_products]` follows descriptor product
order. Export converts voltage samples to FPGA ticks; counts cannot exceed the
frame interval. Scalar valid, packet-loss, RFI and RFI-only counters are
unavailable and set to zero. Those zeros are not measurements. Scalar
loss-fraction metrics are omitted; scalar diagnostic inputs remain synchronized
and range checked.

## Storage and consumers

`N2FrameView` and the Python loader require explicit opt-in for the extension.
Consumers without per-product support, including `N2TimeDownsample`, reject it.
This path ends at the accumulated visibility product and its storage.

`hdf5N2Write` requires a matching `support_mode`. Opt-in `CHORD_0.1` files contain
uint64 `valid_fpga_count_per_product(frequency,product,time)` in FPGA ticks and
explicit support-availability attributes. The eight unavailable scalar
count/fraction datasets are omitted. Scalar output remains `CHORD_0.0`.
Product order and support mode must remain fixed within a file.

Raw files contain a uint32 metadata-size header, serialized metadata and payload
for each frame. They do not embed the descriptor, support mode or product order;
bind them to the exact saved configuration. Record-length checks detect malformed
framing but do not authenticate the layout. Use versioned HDF5 for interchange.

## Tests

`tests/test_n2_per_product_support.py` compares stage output with products and
joint counts computed directly from independent voltage/mask fixtures. Cases
cover tile boundaries, disjoint packet groups, missing inputs, one-sided pairs,
frame rejection, ring reuse and FPGA tick conversion.
`tests/test_n2_per_product_rotation.py` checks fringestopping against an
independent phase calculation. Descriptor and HDF5 tests check layout, product
identity and storage. These are software checks; telescope input mapping,
physical covariance and operational throughput require separate validation.
