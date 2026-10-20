#include "Config.hpp"
#include "DataType.hpp"
#include "N2Util.hpp"
#include "NDArray.hpp"
#include "Stage.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp"
#include "prometheusMetrics.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <functional>
#include <set>
#include <string>
#include <vector>

/**
 * @class DtvRfiMask
 * @brief Fold one 8192- or 16384-sample DTV decision block into the aligned RFImask frame.
 *
 * DTV int8[F] uses 1 = reject; RFImask uint1x8[num_times/1024, F, 128] uses 1 = keep. The output is
 * their intersection: a rejected frequency has every time bit cleared, an accepted one keeps the
 * existing mask. Both the visibility correlator and its packet-loss counter must consume this
 * output. One detector block per RFImask frame; mismatched time or frequency identity stops the
 * run. Optional fine-support rows distinguish invalid rank support from a valid keep.
 * Invalid rows clear every time bit. The permanent-mask limit also clears every bit,
 * independently of the detector score; the raw detector products retain their meaning.
 * A control frequency (the detector's channel 37 noise control) is never masked: its
 * decision and support are checked and recorded upstream, and its SK mask passes through.
 *
 * A missing decision for a required frequency (support (-1,-1), as a degraded detector
 * writes, or no decision frame by the deadline) stops the run unless
 * missing_decision_action is "mask": then the frame is excluded for every required
 * frequency, logged, and counted apart in kotekan_dtvrfimask_missing_decision_frames_total.
 * With a deadline, a decision that arrives after its frame was released is logged and
 * discarded; after three missed deadlines in a row frames are released without waiting
 * until decisions resume.
 *
 * @par Buffers
 * @buffer  rfi_buf     RFImask frames from n2k.
 *         @buffer_format   NDArray uint1x8 [num_times/1024, num_local_freq, 128]
 *         @buffer_metadata chordMetadata
 * @buffer  dtv_buf     DTV decisions from cudaPilotProxyDetector.
 *         @buffer_format   NDArray int8 [num_local_freq]
 *         @buffer_metadata chordMetadata
 * @buffer  out_buf     The combined mask, same format as rfi_buf.
 * @buffer  fine_support_buf Optional int32 [F,2] rows [rank_valid,n_bulk].
 *         (-1,-1) means no fine test; 0/1 means invalid/valid rank support.
 * @buffer  plcounts_in_buf  Optional packet-loss counts from CountLostPLSamplesScalar,
 *         NDArray int32 [1, F], one sub-integration per detector block.
 * @buffer  plcounts_out_buf The same counts with every sample of a missing decision's
 *         frequency counted as lost; feed it to N2Accumulate as in_plcounts_buf.
 * @buffer  rfi_count_buf    The combined mask without the missing-decision exclusions, same
 *         format as rfi_buf; feed it to RfiMaskSum so the RFI count leaves those frames out.
 *         The three lost-sample buffers go together. Without them a missing decision is
 *         excluded through out_buf alone and RfiMaskSum counts it as RFI.
 *
 * @conf    num_local_freq  int  Frequencies per frame.
 * @conf    num_times       int  Samples per detector block; must be 8192 or 16384.
 * @conf    permanent_mask_freq_ids List of unique integer receiver coarse-frequency IDs
 *          in [0,12288) at the reject-all limit (default []).
 * @conf    require_fine_freq_ids List of unique integer receiver coarse-frequency IDs
 *          in [0,12288) that must have an evaluated fine decision (default []).
 *          Requires fine_support_buf when nonempty. Both lists must be disjoint and
 *          contain only IDs in the input frequency assignment.
 * @conf    control_freq_ids List of unique receiver IDs whose decision is never applied
 *          (default []). Disjoint from permanent_mask_freq_ids; on the node.
 * @conf    missing_decision_action String. "stop" (default) or "mask"; "mask" needs
 *          require_fine_freq_ids.
 * @conf    decision_deadline_s Float. Seconds to wait for a decision after its RFImask
 *          frame arrives (default 0: wait without a timeout). With "stop", a missed
 *          deadline stops the run.
 *
 * @par Metrics
 * @metric kotekan_dtvrfimask_missing_decision_frames_total Frames released without a
 *         decision for the required frequencies, by reason (deadline, not_evaluated,
 *         late_discarded counts the late decisions dropped).
 */
class DtvRfiMask : public kotekan::Stage {
public:
    DtvRfiMask(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffers) :
        Stage(config, unique_name, buffers, std::bind(&DtvRfiMask::main_thread, this)),
        frequencies(config.get<int>(unique_name, "num_local_freq")),
        mask_rows(config.get<int>(unique_name, "num_times") / 1024), rfi(get_buffer("rfi_buf")),
        dtv(get_buffer("dtv_buf")), output(get_buffer("out_buf")),
        support(config.exists(unique_name, "fine_support_buf") ? get_buffer("fine_support_buf")
                                                               : nullptr),
        missing_action(
            config.get_default<std::string>(unique_name, "missing_decision_action", "stop")),
        deadline_s(config.get_default<double>(unique_name, "decision_deadline_s", 0.0)),
        missing_frames(kotekan::prometheus::Metrics::instance().add_counter(
            "kotekan_dtvrfimask_missing_decision_frames_total", unique_name, {"reason"})) {
        const int num_times = config.get<int>(unique_name, "num_times");
        if (frequencies <= 0 || (num_times != 8192 && num_times != 16384))
            FATAL_ERROR("DtvRfiMask requires positive num_local_freq and num_times=8192 or 16384");
        rfi->register_consumer(unique_name);
        dtv->register_consumer(unique_name);
        output->register_producer(unique_name);
        const auto read_ids = [&](const char* field, std::vector<int>& result) {
            const auto ids =
                config.get_default<nlohmann::json>(unique_name, field, nlohmann::json::array());
            if (!ids.is_array())
                FATAL_ERROR("DtvRfiMask {:s} must be an array", field);
            for (const auto& id : ids) {
                if (!id.is_number_integer() || id < 0 || id >= 12288)
                    FATAL_ERROR(
                        "DtvRfiMask threshold IDs must be exact nonnegative integers below 12288");
                result.push_back(id.get<int>());
            }
            if (std::set<int>(result.begin(), result.end()).size() != result.size())
                FATAL_ERROR("DtvRfiMask threshold frequency IDs must be unique");
        };
        read_ids("permanent_mask_freq_ids", permanent);
        read_ids("require_fine_freq_ids", required);
        read_ids("control_freq_ids", control);
        for (const int id : required)
            if (std::find(permanent.begin(), permanent.end(), id) != permanent.end())
                FATAL_ERROR("DtvRfiMask finite and permanent-mask thresholds overlap");
        for (const int id : control)
            if (std::find(permanent.begin(), permanent.end(), id) != permanent.end())
                FATAL_ERROR("DtvRfiMask control and permanent-mask frequencies overlap");
        if (!required.empty() && !support)
            FATAL_ERROR("DtvRfiMask required fine thresholds need fine_support_buf");
        if (missing_action != "stop" && missing_action != "mask")
            FATAL_ERROR("DtvRfiMask missing_decision_action must be \"stop\" or \"mask\"");
        if (missing_action == "mask" && required.empty())
            FATAL_ERROR("DtvRfiMask missing_decision_action \"mask\" needs require_fine_freq_ids");
        if (!(deadline_s >= 0.0))
            FATAL_ERROR("DtvRfiMask decision_deadline_s must be nonnegative");
        if (rfi->frame_size != size_t(num_times / 8) * frequencies
            || dtv->frame_size != size_t(frequencies) || output->frame_size != rfi->frame_size)
            FATAL_ERROR("DtvRfiMask buffer sizes do not match one detector block");
        const auto desc = kotekan::GenericNDArray::describe(
            kotekan::uint1x8, "RFImask", {mask_rows, frequencies, 128}, {"T8hi128", "F", "T8lo128"},
            {1024, 1, 8});
        rfi->require_frame_desc(desc);
        output->require_frame_desc(desc);
        dtv->require_frame_desc(kotekan::GenericNDArray::describe(kotekan::int8, "dtv_mask",
                                                                  {frequencies}, {"F"}, {1}));
        if (config.exists(unique_name, "rfi_count_buf")
            || config.exists(unique_name, "plcounts_in_buf")
            || config.exists(unique_name, "plcounts_out_buf")) {
            rfi_count = get_buffer("rfi_count_buf");
            plcounts_in = get_buffer("plcounts_in_buf");
            plcounts_out = get_buffer("plcounts_out_buf");
            rfi_count->register_producer(unique_name);
            plcounts_in->register_consumer(unique_name);
            plcounts_out->register_producer(unique_name);
            if (rfi_count->frame_size != rfi->frame_size
                || plcounts_in->frame_size != size_t(frequencies) * sizeof(int32_t)
                || plcounts_out->frame_size != plcounts_in->frame_size)
                FATAL_ERROR("DtvRfiMask lost-sample buffer sizes do not match one detector block");
            rfi_count->require_frame_desc(desc);
            const auto counts_desc =
                kotekan::GenericNDArray::describe(kotekan::int32, "pl_lost_counts_scalar",
                                                  {1, frequencies}, {"Tc", "F"}, {num_times, 1});
            plcounts_in->require_frame_desc(counts_desc);
            plcounts_out->require_frame_desc(counts_desc);
        }
        if (support) {
            support->register_consumer(unique_name);
            if (support->frame_size != size_t(frequencies) * 2 * sizeof(int32_t))
                FATAL_ERROR("DtvRfiMask fine support buffer size mismatch");
            support->require_frame_desc(kotekan::GenericNDArray::describe(
                kotekan::int32, "dtv_fine_support", {frequencies, 2}, {"F", "S"}, {1, 1}));
        }
    }

    void main_thread() override {
        N2::frameID rfi_id(rfi), dtv_id(dtv), out_id(output);
        N2::frameID support_id(support ? support : dtv);
        N2::frameID rfi_count_id(rfi_count ? rfi_count : output);
        N2::frameID plcounts_in_id(plcounts_in ? plcounts_in : rfi);
        N2::frameID plcounts_out_id(plcounts_out ? plcounts_out : rfi);
        bool started = false;
        int64_t next_seq = 0;
        // Sequence of the oldest frame whose decision is neither applied nor discarded.
        int64_t next_decision_seq = 0;
        int period = 0;
        std::vector<int> bound_freq;
        int missed_in_a_row = 0, discarded_in_a_row = 0;
        bool reported_unevaluated = false;
        while (!stop_thread) {
            const auto* keep = rfi->wait_for_full_frame(unique_name, rfi_id);
            if (!keep)
                break;
            // After repeated misses frames are released at once (deadline now).
            timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            if (missed_in_a_row < FRAMES_BEFORE_RELEASE)
                add_seconds(deadline, deadline_s);
            const auto rm = get_chord_metadata(rfi, rfi_id);
            rm->check_frame_desc(rfi->get_frame_desc<kotekan::GenericNDArray>());
            const auto seq = rm->get_fpga_seq_num();
            const auto rp = rm->get_time_downsampling_fpga();
            if (seq < 0 || rp <= 0 || rp % 1024)
                FATAL_ERROR("DtvRfiMask time alignment mismatch");
            if (!rm->has_coarse_freq() || rm->get_coarse_freq().size() != size_t(frequencies))
                FATAL_ERROR("DtvRfiMask frequency identity mismatch");
            const auto freq = rm->get_coarse_freq();
            if (std::set<int>(freq.begin(), freq.end()).size() != freq.size())
                FATAL_ERROR("DtvRfiMask duplicate frequency identity");
            check_not_upchannelized(rm);
            if (started && (seq != next_seq || rp != period || freq != bound_freq))
                FATAL_ERROR("DtvRfiMask discontinuous time or frequency identity");
            if (!started) {
                for (const auto& ids : {permanent, required, control})
                    for (const int id : ids)
                        if (std::find(freq.begin(), freq.end(), id) == freq.end())
                            FATAL_ERROR("DtvRfiMask threshold frequency ID {:d} is absent", id);
                next_decision_seq = seq;
            }

            const int32_t* plcounts = nullptr;
            if (plcounts_in) {
                // Packet-loss counts come from upstream, not from the detector.
                plcounts = reinterpret_cast<const int32_t*>(
                    plcounts_in->wait_for_full_frame(unique_name, plcounts_in_id));
                if (!plcounts)
                    break;
                const auto pm = get_chord_metadata(plcounts_in, plcounts_in_id);
                pm->check_frame_desc(plcounts_in->get_frame_desc<kotekan::GenericNDArray>());
                if (pm->get_fpga_seq_num() != seq
                    || pm->get_time_downsampling_fpga() != int64_t(rp) * mask_rows
                    || !pm->has_coarse_freq() || pm->get_coarse_freq() != freq)
                    FATAL_ERROR("DtvRfiMask packet-loss counts identity mismatch");
                for (int f = 0; f < frequencies; ++f)
                    if (plcounts[f] < 0 || plcounts[f] > mask_rows * 1024)
                        FATAL_ERROR("DtvRfiMask packet-loss count out of range");
            }

            // Wait for this frame's decision, dropping decisions for frames already released.
            const int8_t* reject = nullptr;
            const int32_t* fine = nullptr;
            int64_t dp = 0;
            while (!reject) {
                const int status = wait_full(dtv, dtv_id, deadline);
                if (status < 0)
                    return;
                if (status > 0)
                    break;
                const auto dm = get_chord_metadata(dtv, dtv_id);
                dm->check_frame_desc(dtv->get_frame_desc<kotekan::GenericNDArray>());
                dp = dm->get_time_downsampling_fpga();
                if (dp <= 0 || int64_t(rp) * mask_rows != dp)
                    FATAL_ERROR("DtvRfiMask time alignment mismatch");
                if (support) {
                    // The support frame follows its decision; give it a fresh deadline.
                    timespec support_deadline;
                    clock_gettime(CLOCK_REALTIME, &support_deadline);
                    add_seconds(support_deadline, deadline_s);
                    const int support_status = wait_full(support, support_id, support_deadline);
                    if (support_status < 0)
                        return;
                    if (support_status > 0)
                        FATAL_ERROR("DtvRfiMask fine support frame missing for the decision at "
                                    "seq {:d}",
                                    dm->get_fpga_seq_num());
                }
                const int64_t decision_seq = dm->get_fpga_seq_num();
                if (deadline_s > 0 && decision_seq < seq && decision_seq == next_decision_seq) {
                    ++discarded_in_a_row;
                    missing_frames.labels({"late_discarded"}).inc();
                    next_decision_seq += dp;
                    dtv->mark_frame_empty(unique_name, dtv_id++);
                    if (support)
                        support->mark_frame_empty(unique_name, support_id++);
                    continue;
                }
                if (decision_seq != seq)
                    FATAL_ERROR("DtvRfiMask time alignment mismatch");
                if (!dm->has_coarse_freq() || dm->get_coarse_freq() != freq)
                    FATAL_ERROR("DtvRfiMask frequency identity mismatch");
                check_not_upchannelized(dm);
                reject = reinterpret_cast<const int8_t*>(dtv->frames[dtv_id]);
                if (support) {
                    const auto sm = get_chord_metadata(support, support_id);
                    sm->check_frame_desc(support->get_frame_desc<kotekan::GenericNDArray>());
                    if (sm->get_fpga_seq_num() != seq || sm->get_time_downsampling_fpga() != dp
                        || !sm->has_coarse_freq() || sm->get_coarse_freq() != freq
                        || !sm->has_freq_upchan_factor() || !sm->has_freq_upchan_index()
                        || sm->get_freq_upchan_factor() != std::vector<int>(frequencies, 1)
                        || sm->get_freq_upchan_index() != std::vector<int>(frequencies, 0))
                        FATAL_ERROR("DtvRfiMask fine support identity mismatch");
                    fine = reinterpret_cast<const int32_t*>(support->frames[support_id]);
                }
            }
            if (discarded_in_a_row > 0) {
                WARN("DtvRfiMask discarded {:d} late decision(s) for frames already released",
                     discarded_in_a_row);
                discarded_in_a_row = 0;
            }

            // rfi_excluded: rejections, invalid rank support and the permanent mask, counted as
            // RFI. missing: required frequencies without a decision, which are invalid.
            std::vector<bool> rfi_excluded(frequencies), missing(frequencies);
            if (!reject) {
                // No decision by the deadline.
                if (missing_action != "mask")
                    FATAL_ERROR("DtvRfiMask decision for seq {:d} missed its {:.3f} s deadline",
                                seq, deadline_s);
                missing_frames.labels({"deadline"}).inc();
                if (++missed_in_a_row == 1)
                    ERROR("DtvRfiMask decision for seq {:d} missed its {:.3f} s deadline: the "
                          "frame is masked for every required frequency",
                          seq, deadline_s);
                if (missed_in_a_row == FRAMES_BEFORE_RELEASE)
                    ERROR("DtvRfiMask {:d} decisions in a row missed their deadline: frames are "
                          "released masked without waiting until decisions resume",
                          missed_in_a_row);
                for (int f = 0; f < frequencies; ++f) {
                    rfi_excluded[f] = listed(permanent, freq[f]);
                    missing[f] = listed(required, freq[f]) && !listed(control, freq[f]);
                }
            } else {
                if (missed_in_a_row > 0)
                    INFO("DtvRfiMask decisions resumed at seq {:d} after {:d} missed deadline(s)",
                         seq, missed_in_a_row);
                missed_in_a_row = 0;
                next_decision_seq = seq + dp;
                bool unevaluated = false;
                for (int f = 0; f < frequencies; ++f) {
                    if (reject[f] != 0 && reject[f] != 1)
                        FATAL_ERROR("DtvRfiMask detector decision must be zero or one");
                    bool invalid = false, unevaluated_row = false;
                    if (fine) {
                        const int32_t valid = fine[2 * f], count = fine[2 * f + 1];
                        if (!((valid == -1 && count == -1)
                              || ((valid == 0 || valid == 1) && count >= 0 && count <= 256
                                  && (valid == 0 || count > 0))))
                            FATAL_ERROR("DtvRfiMask malformed fine support row");
                        unevaluated_row = valid == -1 && listed(required, freq[f]);
                        if (unevaluated_row && missing_action != "mask")
                            FATAL_ERROR("DtvRfiMask required fine threshold was not evaluated");
                        unevaluated = unevaluated || unevaluated_row;
                        invalid = valid == 0;
                    }
                    // A control decision is recorded, never applied.
                    const bool applied = !listed(control, freq[f]);
                    rfi_excluded[f] =
                        applied && (reject[f] || invalid || listed(permanent, freq[f]));
                    missing[f] = applied && unevaluated_row;
                }
                if (unevaluated) {
                    missing_frames.labels({"not_evaluated"}).inc();
                    if (!reported_unevaluated)
                        ERROR("DtvRfiMask required fine threshold not evaluated at seq {:d} (the "
                              "detector is degraded or skipped it): masking those frequencies",
                              seq);
                } else if (reported_unevaluated) {
                    INFO("DtvRfiMask required fine thresholds evaluated again at seq {:d}", seq);
                }
                reported_unevaluated = unevaluated;
            }

            std::vector<bool> excluded(frequencies);
            for (int f = 0; f < frequencies; ++f)
                excluded[f] = rfi_excluded[f] || missing[f];
            if (!write_mask(output, out_id, keep, excluded, rm))
                break;
            if (rfi_count) {
                // The RFI count leaves out the invalid frames; their samples count as lost.
                if (!write_mask(rfi_count, rfi_count_id, keep, rfi_excluded, rm))
                    break;
                auto* lost = reinterpret_cast<int32_t*>(
                    plcounts_out->wait_for_empty_frame(unique_name, plcounts_out_id));
                if (!lost)
                    break;
                std::memcpy(lost, plcounts, plcounts_in->frame_size);
                for (int f = 0; f < frequencies; ++f)
                    if (missing[f])
                        lost[f] = mask_rows * 1024;
                plcounts_out->allocate_new_metadata_object(plcounts_out_id);
                const auto lm = get_chord_metadata(plcounts_out, plcounts_out_id);
                lm->deepCopy(get_chord_metadata(plcounts_in, plcounts_in_id));
                lm->check_frame_desc(plcounts_out->get_frame_desc<kotekan::GenericNDArray>());
                plcounts_in->mark_frame_empty(unique_name, plcounts_in_id++);
                plcounts_out->mark_frame_full(unique_name, plcounts_out_id++);
                rfi_count->mark_frame_full(unique_name, rfi_count_id++);
            }
            started = true;
            next_seq = seq + int64_t(rp) * mask_rows;
            period = rp;
            bound_freq = freq;
            rfi->mark_frame_empty(unique_name, rfi_id++);
            if (reject) {
                dtv->mark_frame_empty(unique_name, dtv_id++);
                if (support)
                    support->mark_frame_empty(unique_name, support_id++);
            }
            output->mark_frame_full(unique_name, out_id++);
        }
    }

private:
    // Missed deadlines in a row after which frames are released without waiting.
    static constexpr int FRAMES_BEFORE_RELEASE = 3;

    static bool listed(const std::vector<int>& ids, const int id) {
        return std::find(ids.begin(), ids.end(), id) != ids.end();
    }

    static void add_seconds(timespec& time, const double seconds) {
        const auto ns = int64_t(seconds * 1e9) + time.tv_nsec;
        time.tv_sec += ns / 1'000'000'000;
        time.tv_nsec = ns % 1'000'000'000;
    }

    void check_not_upchannelized(const std::shared_ptr<chordMetadata>& meta) {
        if (!meta->has_freq_upchan_factor() || !meta->has_freq_upchan_index()
            || meta->get_freq_upchan_factor() != std::vector<int>(frequencies, 1)
            || meta->get_freq_upchan_index() != std::vector<int>(frequencies, 0))
            FATAL_ERROR("DtvRfiMask requires un-upchannelized coarse frequencies");
    }

    /// Write the SK mask with every time bit of the excluded frequencies cleared.
    bool write_mask(Buffer* buffer, N2::frameID& id, const uint8_t* keep,
                    const std::vector<bool>& excluded, const std::shared_ptr<chordMetadata>& rm) {
        auto* frame = buffer->wait_for_empty_frame(unique_name, id);
        if (!frame)
            return false;
        std::memcpy(frame, keep, rfi->frame_size);
        for (int t = 0; t < mask_rows; ++t)
            for (int f = 0; f < frequencies; ++f)
                if (excluded[f])
                    std::memset(frame + (t * frequencies + f) * 128, 0, 128);
        buffer->allocate_new_metadata_object(id);
        const auto meta = get_chord_metadata(buffer, id);
        meta->deepCopy(rm);
        meta->check_frame_desc(buffer->get_frame_desc<kotekan::GenericNDArray>());
        return true;
    }

    /// Wait for a full frame: 0 when it is there, 1 past the deadline, -1 on shutdown.
    /// Without a deadline (decision_deadline_s 0) the wait has no timeout.
    int wait_full(Buffer* buffer, const int id, const timespec& deadline) {
        if (deadline_s <= 0)
            return buffer->wait_for_full_frame(unique_name, id) ? 0 : -1;
        return buffer->wait_for_full_frame_timeout(unique_name, id, deadline);
    }

    const int frequencies;
    const int mask_rows;
    Buffer *rfi, *dtv, *output, *support;
    // Optional lost-sample route: all three or none.
    Buffer *rfi_count = nullptr, *plcounts_in = nullptr, *plcounts_out = nullptr;
    std::vector<int> permanent, required, control;
    const std::string missing_action;
    const double deadline_s;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& missing_frames;
};

REGISTER_KOTEKAN_STAGE(DtvRfiMask);
