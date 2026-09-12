# PilotProxy validation

Build Kotekan with `USE_CUDA=ON`, `WITH_TESTS=ON` and
`PILOTPROXY_EXPORT_BUNDLE=ON`. Install the PilotProxy Python package at the
commit in `external/pilotproxy/VENDOR.json`, along with Kotekan's Python test
dependencies. The CUDA device must support the vendored detector core.

Production configurations select a validated, calibrated CHORD bundle with
`dtv_runtime_bundle_dir`. Bundle export and validation are available from
the `pilot-proxy export-runtime-weight-bundle` and
`pilot-proxy validate-runtime-weight-bundle` commands. Build-time export
creates development inputs; it does not establish deployment calibration.

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

The default pipeline still leaves DTV application disabled. Render `chord.j2`
with `dtv_enabled=true` for detector-only output, or `dtv_apply_mask=true` for
the opt-in applied path. Calibration remains mandatory for pilot channels.
These flags select implementation paths; they do not certify a telescope run.

`DtvRfiMask` intersects one binary DTV rejection per coarse channel with the
existing bit-packed RFI good-sample mask. It supports exactly one 8192-sample
CHORD block per mask frame. It checks start sequence, sample period, coarse
frequency order, un-upchannelized identity, continuity and binary decisions.
An absent input blocks progress; a mismatching input stops the run. It never
reuses a previous decision. The host round trip is intentional in this local
implementation and needs full-load latency/throughput acceptance.

Both `cudaCorrelator` and `cudaPL1bitCorrelator` consume the merged GPU mask;
`RfiMaskSum` consumes the corresponding host mask. Original RFI, DTV decision,
DTV powers and the merged applied mask have distinct recording products. The
existing count consumer receives counts after both DTV/RFI gating and packet
loss; normalizing against nominal time would be incorrect. Zero counts denote
unmeasured products. See the [normalization contract](n2_mask_normalization.md).
The complete GPU-to-accumulator replay remains an acceptance check.

The detector currently masks only each pilot-bearing coarse frequency. It does
not distribute that decision across a 6 MHz television channel or other nodes.
The supported output boundary is the accumulated complex visibility product;
delay filtering and subsequent science analysis are outside this integration.

```sh
python -m pytest tests/test_dtv_rfi_mask.py -q
```

Use the same `PILOTPROXY_TEST_BINARY` and `PILOTPROXY_TEST_BUNDLE` settings as
above. The tests generate local metadata-bearing files, inject known nonuniform
masks and packet loss, and run the adapter, real CUDA visibility/count correlators
and RFI-count stage. They compare all visibility components and counts with CPU
calculations for 128 inputs, four frequencies and eight frames over a four-frame
ring. Both prescribed decisions and the actual synthetically calibrated detector
are exercised. Wrong frequency, late/missing decisions, wrong time periods and
nonbinary values are refused. Configuration tests ensure both correlators and
the RFI-count diagnostic select the same mask and that the default stays off.

These local tests are not Pathfinder operation, scientific threshold calibration,
shared-GPU deadline acceptance, separate reason-coded input-health validation,
or a measured science-transfer function. Shadow and applied telescope runs
remain pending.
