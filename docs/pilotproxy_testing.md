# PilotProxy validation

Build Kotekan with `USE_CUDA=ON`, `WITH_TESTS=ON` and
`PILOTPROXY_EXPORT_BUNDLE=ON`. Install the PilotProxy Python package at the
commit in `external/pilotproxy/VENDOR.json`, along with Kotekan's Python test
dependencies. The CUDA device must support the vendored detector core.

From the repository root, run the complete pipeline tests:

```sh
export PILOTPROXY_TEST_BINARY="$PWD/build/kotekan/kotekan"
export PILOTPROXY_TEST_BUNDLE="$PWD/build/pilotproxy_bundle"
python3 -m pytest -v tests/test_pilotproxy_pipeline.py tests/test_pilotproxy_integration.py
```

The integration tests launch the actual Kotekan binary. They compare every
mask, coarse power and FPGA timestamp against the CPU reference for calibrated
fine mode, mixed fine/coarse operation, coarse fallback and non-pilot channels.
They also cover channel reordering, window reversal, ring wrapping, detector
blocks crossing input-frame boundaries, and malformed bundles and geometry.

Pathfinder uses a padded 64-dish, two-polarization buffer. Sparse-input tests
exercise 16 and 32 live signal inputs distributed across that buffer, as well
as all-zero input. This does not assume how many feeds are currently connected.

For a sustained workload, enable both larger geometries:

```sh
PILOTPROXY_SOAK_FRAMES=2048 python3 -m pytest -v -s \
    tests/test_pilotproxy_integration.py -k production_geometry_soak
```

The soak covers Pathfinder's padded 64 dishes and 384 local channels, then
512 dishes and 48 local channels. It deliberately assigns all 23 pilots to
each test node and populates every input. This exceeds a normal node's pilot
load. Four random voltage frames repeat through the ring; all output frames
are checked against those four independently computed references. Each run
processes 768 GiB for 2048 frames, while saving only four voltage seeds and
the smaller output products. Allow several GiB of temporary disk space.

The GPU Test and Release CI jobs run the integration suite. Release also runs
eight frames per larger geometry; unoptimized input generation makes these
large fixtures much slower in Test builds. Without the two runtime paths, integration tests explicitly skip;
ordinary CPU checks still run. A requested integration run fails if its binary,
bundle or Python reference is missing.

On a Linux GPU host with CUDA debugging support, run the detector sweep under
the memory checker:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 \
    "$PILOTPROXY_TEST_BINARY" -b 127.0.0.1:0 \
    --config config/ci-tests/gpu_batch/test_pilotproxy_detector.yaml
```

All calibrated bundles in these tests are synthetic fixtures. They validate
the implementation, not the scientific thresholds. Deployment still needs
Pathfinder calibration for the actual active inputs, recorded/live data checks,
and a shared-GPU run alongside the correlator and other stages. The standalone
soak is a correctness stress test, not a deployment throughput measurement.

## Applied-mask integration status

The detector and file-pipeline parity tests validate the producer and transport.
The supplied `dtv_chord.j2` include exports the DTV mask/power buffers; it does
not connect them to `cudaCorrelator`'s separate `rfi_RFImask` stream. Applied
masking therefore still requires an adapter and tests of frame ordering,
visibility accumulation and valid-sample/weight normalization. Equal nominal
frame lengths and successful output replay do not demonstrate this connection.
The telescope shadow run and science-transfer measurement remain separate gates.
