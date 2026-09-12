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

#include <algorithm>
#include <bitset>
#include <functional>
#include <limits>
#include <set>
#include <vector>

/**
 * Record packet presence and RFI/DTV rejection per block and eight-input group.
 * Counts use voltage samples. Masks and visibilities are unchanged.
 * Other input health and detector failures are not measured here.
 */
class DtvInputHealth : public kotekan::Stage {
public:
    DtvInputHealth(kotekan::Config& config, const std::string& name,
                   kotekan::bufferContainer& buffers) :
        Stage(config, name, buffers, std::bind(&DtvInputHealth::main_thread, this)),
        nf(config.get<int>(name, "num_local_freq")),
        npol(config.get<int>(name, "num_polarizations")),
        ndish(config.get<int>(name, "num_dishes")), pl(get_buffer("pl_buf")),
        rfi(get_buffer("rfi_buf")), dtv(get_buffer("dtv_buf")), output(get_buffer("out_buf")) {
        if (nf <= 0 || npol <= 0 || ndish <= 0 || ndish % 8
            || config.get<int>(name, "num_times") != 8192)
            FATAL_ERROR("DtvInputHealth requires positive geometry, dishes divisible by 8 and "
                        "num_times=8192");
        pl->register_consumer(unique_name);
        rfi->register_consumer(unique_name);
        dtv->register_consumer(unique_name);
        output->register_producer(unique_name);
        pl->require_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::uint1x8, "pl_mask_exp", {128, nf, npol, ndish / 8, 8},
            {"Thi64", "F", "P", "D8", "Tlo64"}, {64, 1, 1, 8, 8}));
        rfi->require_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::uint1x8, "RFImask", {8, nf, 128}, {"T8hi128", "F", "T8lo128"}, {1024, 1, 8}));
        dtv->require_frame_desc(
            kotekan::GenericNDArray::describe(kotekan::int8, "dtv_mask", {nf}, {"F"}, {1}));
        output->require_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::uint32, "input_health_v1", {1, nf, npol, ndish / 8, 7},
            {"Th", "F", "P", "D8", "Reason"}, {8192, 1, 1, 8, 1}));
    }

    void main_thread() override {
        N2::frameID pi(pl), ri(rfi), di(dtv), oi(output);
        bool started = false;
        int64_t next_seq = 0;
        int block_ticks = 0;
        std::vector<int> frequencies;
        while (!stop_thread) {
            const auto* packet = pl->wait_for_full_frame(unique_name, pi);
            if (!packet)
                break;
            const auto* keep = rfi->wait_for_full_frame(unique_name, ri);
            if (!keep)
                break;
            const auto* reject = dtv->wait_for_full_frame(unique_name, di);
            if (!reject)
                break;
            const auto pm = get_chord_metadata(pl, pi);
            const auto rm = get_chord_metadata(rfi, ri);
            const auto dm = get_chord_metadata(dtv, di);
            if (!pm || !rm || !dm)
                FATAL_ERROR("DtvInputHealth requires CHORD metadata on all streams");
            for (const auto& m : {pm, rm, dm})
                if (!m->has_fpga_seq_num() || !m->has_time_downsampling_fpga()
                    || !m->has_coarse_freq() || !m->has_freq_upchan_factor()
                    || !m->has_freq_upchan_index())
                    FATAL_ERROR("DtvInputHealth missing time/frequency identity");
            const auto seq = dm->get_fpga_seq_num();
            const int period = dm->get_time_downsampling_fpga();
            if (seq < 0 || period <= 0 || period % 8192
                || seq > std::numeric_limits<int64_t>::max() - period
                || pm->get_fpga_seq_num() != seq || rm->get_fpga_seq_num() != seq
                || int64_t(pm->get_time_downsampling_fpga()) * 128 != period
                || int64_t(rm->get_time_downsampling_fpga()) * 8 != period)
                FATAL_ERROR("DtvInputHealth time identity mismatch");
            const auto freq = dm->get_coarse_freq();
            if (freq.size() != size_t(nf)
                || std::set<int>(freq.begin(), freq.end()).size() != freq.size())
                FATAL_ERROR("DtvInputHealth invalid frequency identity");
            for (const auto& m : {pm, rm, dm})
                if (m->get_coarse_freq() != freq
                    || m->get_freq_upchan_factor() != std::vector<int>(nf, 1)
                    || m->get_freq_upchan_index() != std::vector<int>(nf, 0))
                    FATAL_ERROR("DtvInputHealth frequency identity mismatch");
            if (started && (seq != next_seq || period != block_ticks || freq != frequencies))
                FATAL_ERROR("DtvInputHealth discontinuous stream identity");
            for (int f = 0; f < nf; ++f)
                if (reject[f] > 1)
                    FATAL_ERROR("DtvInputHealth decision must be 0 or 1");
            auto* raw = output->wait_for_empty_frame(unique_name, oi);
            if (!raw)
                break;
            auto* count = reinterpret_cast<uint32_t*>(raw);
            const size_t ng = size_t(npol) * (ndish / 8);
            std::fill(count, count + size_t(nf) * ng * 7, 0u);
            for (int f = 0; f < nf; ++f) {
                for (size_t g = 0; g < ng; ++g) {
                    auto* c = count + (size_t(f) * ng + g) * 7;
                    for (int byte_t = 0; byte_t < 1024; ++byte_t) {
                        const size_t pl_index =
                            ((size_t(byte_t / 8) * nf + f) * ng + g) * 8 + byte_t % 8;
                        const size_t rfi_index =
                            (size_t(byte_t / 128) * nf + f) * 128 + byte_t % 128;
                        const uint8_t present = packet[pl_index];
                        const uint8_t rfi_bad = present & uint8_t(~keep[rfi_index]);
                        const uint32_t n = std::bitset<8>(present).count();
                        const uint32_t nr = std::bitset<8>(rfi_bad).count();
                        c[0] += n;
                        c[1] += 8 - n;
                        c[2] += nr;
                        c[3] += reject[f] ? n : 0;
                        c[4] += reject[f] ? nr : 0;
                        c[5] += reject[f] ? 0 : n - nr;
                    }
                    c[6] = 1; // other-health-unknown, not a good/bad science decision
                }
            }
            output->allocate_new_metadata_object(oi);
            const auto om = get_chord_metadata(output, oi);
            om->deepCopy(dm);
            om->set_from_frame_desc(output->get_frame_desc<kotekan::GenericNDArray>());
            om->set_name("input_health_v1");
            started = true;
            next_seq = seq + period;
            block_ticks = period;
            frequencies = freq;
            pl->mark_frame_empty(unique_name, pi++);
            rfi->mark_frame_empty(unique_name, ri++);
            dtv->mark_frame_empty(unique_name, di++);
            output->mark_frame_full(unique_name, oi++);
        }
    }

private:
    const int nf, npol, ndish;
    Buffer *pl, *rfi, *dtv, *output;
};

REGISTER_KOTEKAN_STAGE(DtvInputHealth);
