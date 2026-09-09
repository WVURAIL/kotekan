# Packet/RFI count geometry and native product ordering

`cudaPL1bitCorrelator` now refuses unsupported count-kernel geometry at construction. Previously, `Sds` values other than 16 or 128 emitted an error once, then filled every count with the nominal subintegration length. That fallback ignored both packet loss and the RFI mask. Its full counts passed ordinary range/scalar-equality checks, so downstream normalization could silently treat missing support as valid.

The shared GPU-free `n2_count_station_groups(P,D)` guard in `lib/utils/n2CountGeometry.hpp` requires positive polarizations/dishes, dishes divisible by eight, a representable station-group product, and `P*(D/8)` equal to 16 or 128. For two polarizations this means 64 or 512 dishes (128 or 1024 voltage inputs). The checked value is initialized before this command's buffer objects and stream registration. Unsupported geometry produces no count output. Accepted geometry calls the same native kernel as before; its remaining time/frequency/divisibility checks are unchanged. This guard is not a new kernel implementation or a guarantee that an entire pipeline configuration is supported.

The station and product mapping is independently supported by these native sources:

- `cudaCorrelator.cpp` supplies contiguous voltage `[T,F,P,D]` as n2k `[T,F,S]`. `cudaPL1bitCorrelator.cpp` supplies `[T/64,F,P,D/8]` as `[T/64,F,Sds]`; a packet group is therefore `floor(i/8)` for native flattened station index `i`.
- `external/n2k/include/n2k/Correlator.hpp` defines `V_ij = sum E_i conj(E_j)` and the blocked-16 offset. In complex-product units it is `256*triangle(floor(i/16),floor(j/16)) + 16*(i%16) + j%16`.
- `external/n2k/src_lib/pl_1bit_correlator.cu` uses `mma_b1_m8_n8_k128`. The implementation in `external/ksgpu/include/ksgpu/device_mma.hpp` is `and.popc`, so counts are intersections after the shared RFI mask, not minima/averages of marginal counts.
- The PL kernel's `write_8_8` register/lane assignment makes each 8-by-8 tile row-major. Both supported kernels store lower-triangular tiles. The count index is consequently `64*triangle(floor(g_i/8),floor(g_j/8)) + 8*(g_i%8) + g_j%8`, where `g_i=floor(i/8)`.
- Fringestopping multiplies native `V_ij` by `phase_i*conj(phase_j)`. Conjugating once when writing upper-triangular `N2::cmap(j,i)` preserves the stated native convention.

Independent integer enumeration checks all 8,704 stored entries in the Sds=128 kernel write schedule, with no duplicates or holes, and all 8,256 / 524,800 science-product mappings for 128 / 1024 inputs. This is source-layout evidence, not a GPU execution or a measurement of physical support.

The current per-product accumulation review finds the `N*k/Q` implementation consistent with admitted positive-count pairs. Zero-count raw products are excluded before multiplication; one-sided pairs add visibility support but no difference; rejected frame pairs add neither; output count units convert samples to FPGA ticks; all product sums/counts/differences reset between output bins. Its statistical interpretation still assumes common per-sample error variance and equal expected normalized visibility within each pair. Precision is unavailable when support, usable differences or positive finite Q are absent. No new physical calibration follows from this normalization.

Validation uses the production guard in a standalone host C++ executable via `tests/test_n2_count_geometry.py`: four accepted geometries and twelve unsupported/malformed/overflow cases. The isolated command compiles without CUDA, and the actual changed CUDA command also compiles as a host C++ object using the existing CUDA release build's flags. No GPU code is executed and no shared build directory is changed. The analysis virtual environment lacks the general test suite's `pytest_localserver` dependency, so the isolated invocation is:

```sh
python -m pytest --noconftest -q tests/test_n2_count_geometry.py
```

This test uses no project conftest fixtures. The recorded run passed all 16 cases. A full pipeline/GPU launch test remains separate. Evidence, source hashes, frozen-campaign separation checks, compiler commands and independent mapping script are under `output/pathfinder-per-product-support-2026-09-09/producer-guard/` in the workspace for inclusion in the per-product release.
