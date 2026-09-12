# Packet/RFI count geometry

`cudaPL1bitCorrelator` requires positive polarization and dish counts, dishes
divisible by eight, and `num_polarizations * (num_dishes / 8)` equal to 16 or
128. For two polarizations this means 64 or 512 dishes. The constructor rejects
unsupported geometry before registering buffers; it never substitutes nominal
counts for unavailable kernel support. Other native kernel constraints still
apply.

The shared host-only guard is `n2_count_station_groups` in
`lib/utils/n2CountGeometry.hpp`. Run its accepted-geometry, malformed-input and
overflow tests with:

```sh
python -m pytest --noconftest -q tests/test_n2_count_geometry.py
```

Native station order flattens `[polarization, dish]`, with packet groups of
eight adjacent dishes. The count kernel intersects both packet masks with the
common RFI mask. Its lower-triangular 8-by-8 tiles are row-major; correlation
tiles use block size 16. See [per-product support](n2_per_product_support.md)
for the index mapping. CPU mapping tests check software layout; GPU execution
and physical input mapping require separate validation.
