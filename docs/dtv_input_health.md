# DTV mask and packet-loss recording

`DtvInputHealth` records packet presence and RFI/DTV rejections for each
8192-sample block without changing the masks or visibilities. Render `chord.j2`
with `dtv_enabled` and `dtv_record_input_health`; `dtv_input_health_dir` sets the
HDF5 output directory. When `dtv_apply_mask` is false, kept counts describe
what would survive masking. Save that setting with the recorded configuration.

The `input_health_v1` uint32 array has shape
`[1, frequency, polarization, dish_group, 7]`. Group `g` covers dish positions
`8*g..8*g+7` within one polarization in the producer's input order.
Physical station identities must be supplied separately.

| Column | Meaning |
|---|---|
| 0 | Present voltage samples |
| 1 | Missing voltage samples |
| 2 | Present samples rejected by the original RFI mask |
| 3 | Present samples rejected by the DTV decision |
| 4 | Present samples rejected by both masks |
| 5 | Present samples kept by both masks |
| 6 | Other input health unknown (`1`), a status bit |

Columns 0-5 count voltage samples. Metadata records the FPGA start tick and
block span. Each group obeys `present + missing = 8192` and
`missing + rfi_rejected + dtv_rejected - overlap + kept = 8192`.
Subtracting the overlap counts samples rejected by both masks only once.

Inputs must have matching frequency and time coordinates, a fixed integer
sample period, consecutive frames and binary decisions. Malformed input stops
the stage; missing input can block it. This recorder does not measure clipping,
gain errors or detector failures, so zero rejections do not imply healthy data.
Group totals cannot recover the joint sample counts for each baseline.

Run the CPU tests with:

```sh
KOTEKAN_BUILD_DIRNAME=build python3 -m pytest -q tests/test_dtv_input_health.py
```

Tests compare the output with sample-level Boolean masks and cover ring reuse,
FPGA tick conversion, invalid inputs and template settings. Sustained throughput
and hardware input identities still need to be checked on the telescope.
