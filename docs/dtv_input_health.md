# DTV mask and packet-loss health recording

`DtvInputHealth` is an optional CPU observer of the original RFI mask, expanded
packet-presence mask and binary DTV decision. It uses the same 8192-sample
blocks as `DtvRfiMask` and records the native input-group axis, frequency
identity and FPGA interval. The science buffers and mask decisions are not
modified. The first implementation is for local replay; sustained full-load
throughput and hardware input identities require separate acceptance.

Enable `dtv_enabled` and `dtv_record_input_health` when rendering `chord.j2` to
attach the recorder and its HDF5 sink. `dtv_input_health_dir` selects the output
directory. Recording can run in shadow mode: its kept count is the prospective
DTV/RFI intersection, not proof that the visibility consumer applied that mask.
The recorded config must retain the separate `dtv_apply_mask` setting.

The output is a `uint32` ndarray named `input_health_v1` with shape
`[1, frequency, polarization, dish_group, 7]`. `dish_group=g` covers logical
dish positions `8*g..8*g+7` in the producer's polarization/dish ordering. Each
packet-presence bit describes those eight inputs. This is not a measured
physical station map; map identity and validity must accompany real data.

| Reason column | Meaning |
|---|---|
| 0 | Present voltage samples |
| 1 | Missing voltage samples from the packet-presence mask |
| 2 | Present samples rejected by the original RFI mask |
| 3 | Present samples rejected by the DTV block decision |
| 4 | Present samples rejected by both RFI and DTV |
| 5 | Present samples kept by both masks |
| 6 | Other input health unknown (`1`); this is a status bit, not a count |

Columns 0..5 count voltage samples. The FPGA start and downsampling metadata
specify the half-open interval `[start, start+8192*period)`, in FPGA ticks.
Sample period and frequency order are fixed across frames. Each record obeys
`present + missing = 8192` and
`missing + rfi_rejected + dtv_rejected - overlap + kept = 8192`.
The overlap column prevents double-counting without assigning an arbitrary
priority to concurrent causes. Missing samples cannot be called RFI detections.

The stage requires matching metadata, native coarse frequencies, positive
integer sample periods, consecutive frames and binary DTV decisions. Missing,
late, reordered or malformed inputs are refused; a saved previous decision is
never substituted. A stream that stops producing can block pending its input;
a live watchdog/deadline and durable execution-failure events remain separate.

`other_health_unknown=1` is deliberate. A zero DTV decision can also mean an
unmonitored/disabled channel in the current producer. These streams do not
carry calibrated detector-execution validity, gain state, clipping thresholds,
physical input mapping or a reason for an upstream bad-input flag. Consequently
zero rejection counts do not certify healthy inputs, a clean channel or valid
science data. This is the mask/loss portion of the broader reason-coded health
surface, with the missing portions explicitly represented as unknown.

The CPU test uses independently constructed sample-level Boolean masks before
packing. It checks every emitted count across eight frames and ring reuse,
including wholly missing groups/frames, overlapping rejection causes, packing
boundaries, all-accept/all-reject decisions, and one/four FPGA ticks per sample.
Malformed identity tests exercise refusal paths. Run in a CPU build:

```sh
KOTEKAN_BUILD_DIRNAME=build \
python3 -m pytest -q tests/test_dtv_input_health.py
```

The new sidecar cannot supply per-baseline joint counts by combining its
single-group counts: two groups can lose different samples with equal totals.
Actual per-baseline support must use their sample-by-sample intersection.
