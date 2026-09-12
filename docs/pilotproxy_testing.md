# PilotProxy validation

Build Kotekan with `USE_CUDA=ON` and `WITH_TESTS=ON`, and install its Python test
dependencies. The CUDA device must support the vendored detector core.
The runtime bundle is a separate input. From the repository root, install the
matching PilotProxy package, export the CHORD bundle and run the tests:

```sh
PP_COMMIT="$(python3 -c 'import json; print(json.load(open("external/pilotproxy/VENDOR.json"))["upstream_commit"])')"
python3 -m pip install "git+https://github.com/WVURAIL/pilot-proxy@${PP_COMMIT}"
PP_CONFIGS="$(python3 -c 'from pilot_proxy.paths import CONFIGS_DIR; print(CONFIGS_DIR)')"
export PILOTPROXY_TEST_BINARY="$PWD/build/kotekan/kotekan"
export PILOTPROXY_TEST_BUNDLE="$PWD/build/pilotproxy_bundle"
pilot-proxy export-runtime-weight-bundle \
    --receiver-profile "$PP_CONFIGS/receiver_profiles/chord_dtv_fengine.json" \
    --detector-core-profile "$PP_CONFIGS/detector_core/pilotproxy_cuda_local_reference_power_ratio.json" \
    --weight-coordinate-system post_spectral_sense_normalization \
    --physical-channel-range 14:36 --output-dir "$PILOTPROXY_TEST_BUNDLE"
pilot-proxy validate-runtime-weight-bundle --bundle-dir "$PILOTPROXY_TEST_BUNDLE"
python3 -m pytest -v tests/test_pilotproxy_pipeline.py \
    tests/test_pilotproxy_integration.py tests/test_dtv_rfi_mask.py
```

For telescope runs, set `dtv_runtime_bundle_dir` to a CHORD bundle calibrated
for the active inputs. The exported development bundle is for testing.

The integration tests run Kotekan and compare every
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
large fixtures much slower in Test builds. Without the two runtime paths,
integration tests explicitly skip;
ordinary CPU checks still run. A requested integration run fails if its binary,
bundle or Python reference is missing.

On a Linux GPU host with CUDA debugging support, run the detector sweep under
the memory checker:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 \
    "$PILOTPROXY_TEST_BINARY" -b 127.0.0.1:0 \
    --config config/ci-tests/gpu_batch/test_pilotproxy_detector.yaml
```

These tests use synthetic calibration. Before deployment, calibrate thresholds
for the active inputs, check recorded and live data, and measure throughput
with the detector and correlator sharing the GPU. The standalone soak checks
correctness under sustained load; it does not measure the full pipeline.

## Applying the DTV mask

DTV detection and masking are disabled by default. Render `chord.j2` with
`dtv_enabled=true` to record detector output. Also set `dtv_apply_mask=true`
to apply its decisions. Both paths require calibrated pilot channels.

`DtvRfiMask` intersects one binary DTV rejection per coarse channel with the
existing bit-packed RFI good-sample mask. It supports exactly one 8192-sample
CHORD block per mask frame. It checks start sequence, sample period, coarse
frequency order, absence of upchannelization, continuity and binary decisions.
Missing input blocks processing, and mismatched input stops the run. Previous
decisions are not reused. The mask passes through host memory, so its latency
must be measured with the full pipeline running.

Both `cudaCorrelator` and `cudaPL1bitCorrelator` consume the merged GPU mask;
`RfiMaskSum` consumes the corresponding host mask. Original RFI, DTV decision,
DTV powers and the merged applied mask have distinct recording products. The
existing count consumer receives counts after both DTV/RFI gating and packet
loss; normalizing against nominal time would be incorrect. Zero counts denote
unmeasured products. The [scalar accumulation fixes](n2_mask_normalization.md)
are a prerequisite for enabling the applied path. The tests check raw
correlator visibilities and counts. Replay through the GPU and accumulator
together remains to be checked.

The detector currently masks only each pilot-bearing coarse frequency. It does
not distribute that decision across a 6 MHz television channel or other nodes.
This integration ends at accumulated complex visibility output.

```sh
python -m pytest tests/test_dtv_rfi_mask.py -q
```

Use the same `PILOTPROXY_TEST_BINARY` and `PILOTPROXY_TEST_BUNDLE` settings as
above. The tests generate local metadata-bearing files, inject known nonuniform
masks and packet loss, and run the adapter, CUDA visibility and count correlators,
and RFI-count stage. They compare all visibility components and counts with CPU
calculations for 128 inputs, four frequencies and eight frames over a four-frame
ring. They exercise prescribed decisions and detector output with synthetic
calibration. Cases include wrong frequencies, late or missing decisions,
incorrect sample periods and nonbinary values. Configuration tests ensure both correlators and
the RFI-count diagnostic select the same mask and that the default stays off.

Telescope runs with recording only and with masking applied are still needed,
including measurements of the effect on the accumulated visibilities.
