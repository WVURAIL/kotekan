.. _pilotproxy:

********************************
DTV Pilot Detection (PilotProxy)
********************************

PilotProxy detects ATSC digital television (DTV) on a CHORD GPU node, next to
the n2k correlator, and can mask the affected coarse frequency. It is off by
default.

An 8-VSB DTV channel is 6 MHz wide and carries a steady pilot tone
309.441 kHz above its lower edge. For each ATSC channel from 14 to 36 (470 to
608 MHz) on a node, the stage watches the coarse frequency that holds the
pilot and decides once per 8192-sample block (41.94304 ms, one n2k
integration) whether the pilot is there.

The tests are described in :ref:`dev_pilotproxy_testing`.


Components
==========

``cudaPilotProxyDetector`` (``lib/cuda``)
    A ``cudaCommand`` that reads the same voltage ring as the n2k correlator.
    For each bound frequency it packs one block from every input into 128
    windows of 64 samples and runs the detector core. Per local frequency and
    block it writes ``dtv_mask`` (``int8[F]``, 1 rejects the block),
    ``dtv_powers`` (``uint64[F, 3]``, the target and the two reference frame
    powers, each the sum of that term's 256 fine powers) and, optionally,
    ``dtv_fine_support`` (``int32[F, 2]``, see below).
    Frequencies without a pilot get zero masks and powers. The detector sums
    every input in the ring unless ``dtv_exclude_bad_inputs`` is set (see
    `Bad inputs`_).

``external/pilotproxy``
    The CUDA detector core, copied unchanged from the PilotProxy project.
    ``VENDOR.json`` pins the upstream revision and the hash of each file, and
    ``tools/check_vendored_pilotproxy.py`` checks them. Changes to the core are
    made upstream and re-vendored, not edited here.

``DtvRfiMask`` (``lib/stages``)
    Combines the DTV decision with the SK mask ``rfi_RFImask``. A rejected
    frequency has every sample of the block cleared; a kept one keeps its SK
    mask. The result, ``dtv_RFImask``, replaces ``rfi_RFImask`` for n2k
    correlation, n2k counts and ``RfiMaskSum``. A control frequency (see
    `Channel 37 control`_) is never masked.

For each of 256 fine bins the core forms the ratio of the power in a target
cell at the pilot to the power in two reference cells beside it. The block is
rejected when a bin at the pilot exceeds a calibrated multiple of a rank
statistic of that ratio over the bundle's bulk bins. The arithmetic is exact
integer arithmetic, and the GPU result matches the CPU reference bit for bit.
``dtv_fine_support`` says whether each test was valid: ``(1, n)`` is a valid
test using ``n`` bulk bins; ``(0, n)`` means too few usable bulk bins, with mask
0, which is not a pass; ``(-1, -1)`` means no test was run (no pilot,
permanently masked, or detector disabled). It does not show packet loss, input
health or calibration quality.


Runtime bundle
==============

``dtv_runtime_bundle_dir`` names a directory holding ``pilot_profiles.json``,
``weights.bin``, ``weights.manifest.json``, ``detector_contract.json`` and
``sha256sums.txt``, exported with the PilotProxy Python tools, which are not
part of Kotekan. Each profile gives the ATSC channel, the receiver
``coarse_freq`` ID it binds to (``chord_channel_id``), its weights, and its
fine calibration, whose status is ``pending_campaign`` (not calibrated),
``calibrated``, or ``control`` (channel 37 only, see `Channel 37 control`_).
An optional ``marker_guard_bins`` in the fine calibration widens the bins
next to the designated set that the bulk must leave out: the stage refuses a
bulk with a bin within ``designated_half_width + marker_guard_bins`` (at least
two) of the anchor. It defaults to 0.

At startup the stage checks the bundle's digests as the PilotProxy validator
does: ``sha256sums.txt`` must list the other four files with their SHA-256,
and ``weights_sha256`` in ``pilot_profiles.json`` and
``weights.manifest.json`` must match ``weights.bin``. This needs Kotekan built
with OpenSSL.

The stage checks the bundle's structure at startup, but not the quality of the
calibration. Each node running the stage loads the bundle and creates its
detector handles at startup, even a node with no pilot. Channels are bound from
the first voltage frame's ``coarse_freq`` metadata and stay fixed for the run;
a new bundle needs a restart.


Configuration
=============

``config/fengine/chord.j2`` includes ``include/dtv_chord.j2`` when either
switch is true. Both default to false.

``dtv_enabled: true`` (record only)
    The detector runs and its mask and powers are written to HDF5. n2k still
    reads the SK mask, so the visibilities do not change.

``dtv_apply_mask: true``
    ``DtvRfiMask`` builds ``dtv_RFImask`` and n2k reads it, so a rejected block
    leaves the visibilities and the valid-sample counts together. A frequency
    with support ``(0, n)`` is also cleared for that block. The combined mask is
    formed on the host and copied back to the GPU, and is written to HDF5.

Other keys:

``dtv_runtime_bundle_dir``
    The bundle directory (default ``data/pilotproxy_bundle``).

``dtv_permanent_mask_freq_ids``
    Receiver IDs that the detector skips. With ``dtv_apply_mask`` every sample
    of these frequencies is cleared; in record-only mode they are just not
    evaluated.

``dtv_require_fine_freq_ids``
    Receiver IDs that must have a fine test in every block; a row of
    ``(-1, -1)`` stops the run instead of counting as a pass. ``DtvRfiMask``
    enforces this, so it applies only with ``dtv_apply_mask``.

``dtv_control_freq_ids``
    Receiver IDs of the channel 37 noise control, passed to both stages (see
    `Channel 37 control`_).

``dtv_exclude_bad_inputs``
    Exclude the inputs the bad feed mask flags (see `Bad inputs`_). Default
    false.

``dtv_runtime_failure_mode``, ``dtv_missing_decision_action``, ``dtv_decision_deadline_s``
    What a run-time detector failure or a missing decision does (see
    `What stops the run`_). The defaults, ``stop``, ``stop`` and 0, stop the
    run as before.

``dtv_decouple_detector``, ``dtv_detach_timeout_s``
    Give the detector its own detachable copy of the voltages (see
    `What stops the run`_). Default false.

These lists hold receiver IDs (``coarse_freq``), not ATSC channel numbers. They
must not overlap, and every listed ID must be on the node, so they are set per
node.


Channel 37 control
==================

ATSC channel 37 (608 to 614 MHz) is reserved for radio astronomy and carries
no DTV, so the detector's decisions there measure its false-alarm rate on the
sky. A bundle profile for physical channel 37 with fine calibration status
``control`` is bound and evaluated like a calibrated one: its decision goes to
``dtv_mask``, its frame powers to ``dtv_powers`` and its support to
``dtv_fine_support``. ``DtvRfiMask`` never applies it. Only channel 37 can be
the control, and channel 37 is never ``calibrated``.

List the control's receiver ID in ``dtv_control_freq_ids``; the template
passes it to both stages. The detector stops at the first frame if a bound
control profile is not listed or a listed ID binds no control profile, and
``DtvRfiMask`` refuses a control ID that is also in
``dtv_permanent_mask_freq_ids``. Every other channel the bundle does not
calibrate is refused as before.


Bad inputs
==========

With ``dtv_exclude_bad_inputs`` the detector reads the bad feed mask ring
(``bf_mask``, written by ``bufferBadInputs``, 1 = good) that the SK flagger
uses, with the same claim rule as ``cudaRFIS012tilde``. The packer writes a
flagged input's samples as zeros. A zero input row has zero row sums and zero
fine powers, and the frame and fine powers are sums over inputs, so every
output equals that of the detector run on the good inputs alone; a faulty feed
cannot hold a channel masked. The vendored core is unchanged.
``bf_mask_lifetime_in_samples`` must be a multiple of the detector block and the
mask stream must start at the voltage stream's first sequence number; the run
stops otherwise. Without the switch every input is summed, as before.


GPU streams
===========

Every ``cudaProcess`` on a GPU shares that GPU's CUDA streams. The F-engine and
n2k stages use the defaults: copies to the GPU on stream 0, copies to the host
on stream 1 and kernels on stream 2. ``dtv_chord.j2`` gives the detector stage
``num_cuda_streams: 5``, runs its kernels on stream 3 and copies its products to
the host on stream 4, so the detector never queues behind the correlator, and
the correlator never queues behind it. The streams let the two overlap; they
still share the GPU's cores and memory bandwidth, at the same priority.

The voltage ring orders the two stages: the detector reads a block only after
the copy that wrote it has finished on the GPU, and the block is overwritten
only after the detector's last copy of it has finished. In record-only mode n2k
never waits for the detector. With ``dtv_apply_mask`` n2k reads each block's
``dtv_RFImask``, so it waits for that block's detector run, the host combine
and the copy back (on stream 0, like the other copies to the GPU).


Pilot frequency only
====================

The stage masks only the coarse frequency that holds each pilot. A DTV channel
spans about 30 CHORD coarse frequencies, spread across GPU nodes, and nothing
carries a pilot decision to the others. To remove a whole DTV channel, set
``dtv_apply_mask`` and list its receiver IDs in each node's
``dtv_permanent_mask_freq_ids``.


What stops the run
==================

These stop the whole Kotekan instance, the correlator included:

* a bundle that fails its digest checks, a malformed bundle or a malformed
  configuration, at startup;
* at the first frame: missing or miscounted ``coarse_freq`` metadata, a listed
  receiver ID that is not on the node, or a bound channel that is not
  calibrated;
* a ``coarse_freq`` change after binding, or a CUDA or detector error, unless
  ``dtv_runtime_failure_mode`` is ``degrade`` (below);
* in ``DtvRfiMask``: mismatched timestamps, frequencies or layouts, a malformed
  support row, or an unevaluated row for a required frequency, unless
  ``dtv_missing_decision_action`` is ``mask`` (below).

Degraded mode. With ``dtv_runtime_failure_mode: degrade`` a run-time error
from the detector (a failed packer launch, an error the core reports, a
failed mask copy) is logged at ERROR and the detector stops evaluating for the
rest of the run: that frame and every later one report mask 0, powers 0 and
support ``(-1, -1)`` for every channel, so no failure becomes a clean decision,
and the correlator keeps running. An error that leaves the CUDA context
unusable still stops the run, as do all startup refusals. ``degrade`` needs
the fine support product. The ``fault_inject_frame`` key of
``cudaPilotProxyDetector`` makes the core refuse one call at that frame, for
commissioning tests only.

Missing decisions. With ``dtv_missing_decision_action: mask``, ``DtvRfiMask``
treats a ``(-1, -1)`` row for a required frequency as a missing decision: the
frame is excluded for that frequency. With ``dtv_decision_deadline_s`` above
zero it waits that long for each decision; at the deadline it releases the
frame excluded for every required frequency (``stop`` stops the run instead),
logs and discards the decision if it arrives later, and after three missed
deadlines in a row releases frames without waiting until decisions resume.
Frames excluded this way are counted in
``kotekan_dtvrfimask_missing_decision_frames_total`` (label ``reason``:
``not_evaluated``, ``deadline``, or ``late_discarded`` for dropped late
decisions).

Lost, not RFI. With ``dtv_apply_mask`` and ``dtv_missing_decision_action:
mask``, ``dtv_chord.j2`` routes a frame without a decision to the packet-loss
count N2Accumulate already keeps: ``DtvRfiMask`` reads the n2k packet-loss
counts and writes them with every sample of that frequency's frame counted as
lost (``host_dtv_plcounts_buffer``, N2Accumulate's ``in_plcounts_buf``), and
writes a second mask without those exclusions (``host_dtv_rfi_count_buffer``)
for ``RfiMaskSum``. The frame still leaves the visibilities and the valid
counts through ``dtv_RFImask``, so in the N2 metadata its samples appear in
``n_pl_fpga_ticks``, not in ``n_rfi_fpga_ticks`` or ``n_rfi_only_fpga_ticks``.
This needs one n2k sub-integration per detector block, as on CHORD
(``n2k_sub_integration_ntime`` 8192); ``DtvRfiMask`` stops at startup
otherwise. Without these buffers a frame without a decision is excluded through
``dtv_RFImask`` alone and counted as RFI.

A stalled detector. The detector reads the correlator's voltage ring, so a
detector frame that never completes holds that ring and stalls the
correlator. With ``dtv_decouple_detector`` the detector reads its own copy of
the voltages instead: ``bufferCopy`` copies each voltage frame into
``host_dtv_voltage_buffer`` and, when that buffer has stayed full for
``dtv_detach_timeout_s`` (default 1 s), detaches it for the rest of the run,
so the correlator's input is held at most that long once. The decision
deadline then releases every later frame without a decision, as above; set
``dtv_decision_deadline_s``. The copy costs a second host voltage buffer, a
second GPU voltage ring and its host-to-GPU copy. It does not cover a detector
call that blocks on the host inside ``execute``: the GPU processes of a device
queue their commands under one lock, so that stalls every GPU process on the
device. The ``fault_hang_frame`` and ``fault_hang_seconds`` keys hold one
frame's voltage claim, for commissioning tests only.

So an uncalibrated channel has to be left out of the bundle, or listed in
``dtv_permanent_mask_freq_ids``, before the run starts. A node with no bundle
pilot runs with the detector disabled and writes all-zero masks, which means
"not tested", not clean data.


Monitoring
==========

* At the first frame the log has a ``bound ATSC physical channel`` line for
  each pilot, a ``reject-all limit`` line for each permanently masked ID, or a
  ``detector DISABLED`` line. Check them against the node's frequency map.
* ``output_chord.j2`` writes ``dtv_mask`` and ``dtv_powers``, and with
  ``dtv_apply_mask`` also ``dtv_RFImask``. The ``RfiMaskSum`` counts then
  include the DTV rejections.
* ``dtv_powers`` follows the pilot and reference powers in every block, so a
  moved pilot or a new line near a reference bin can show up there even when
  the mask does not fire. Either is a reason to recalibrate.
