#include "N2TimeDownsample.hpp"

#include "Config.hpp"
#include "FrameDesc.hpp"
#include "Hash.hpp"
#include "N2FrameDesc.hpp"
#include "N2FrameView.hpp"
#include "N2Layout.hpp"
#include "N2Util.hpp"
#include "StageFactory.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "kotekanLogging.hpp"
#include "prometheusMetrics.hpp"
#include "timeUtil.hpp"

#include "fmt.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;
using N2::frameID;

REGISTER_KOTEKAN_STAGE(N2TimeDownsample);

N2TimeDownsample::N2TimeDownsample(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2TimeDownsample::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    const auto in_desc = in_buf->get_frame_desc();
    const auto out_desc = out_buf->get_frame_desc();
    if (in_buf->buffer_type != "N2" || out_buf->buffer_type != "N2" || !in_desc || !out_desc
        || *in_desc != *out_desc)
        FATAL_ERROR("N2TimeDownsample requires matching N2 frame descriptors");

    num_bins_per_rotation = config.get<uint32_t>(unique_name, "num_bins_per_rotation");
    max_age = config.get_default<float>(unique_name, "max_age", 120.0);
    do_fringestop = config.get_default<bool>(unique_name, "do_fringestop", true);
    num_elements = config.get<size_t>(unique_name, "num_elements");
    nprod = 0;
    num_eigenvectors = 0;
    if (num_bins_per_rotation == 0 || !std::isfinite(max_age) || max_age <= 0 || num_elements == 0)
        FATAL_ERROR("N2TimeDownsample requires positive finite bin count, max_age and input count");
    feed_positions_m = Telescope::instance().get_feed_positions_m(
        num_elements, Telescope::instance().fiducial_element_order());
}

void N2TimeDownsample::main_thread() {
    frameID frame_id(in_buf), output_frame_id(out_buf);
    const Telescope& tel = Telescope::instance();
    const bool per_product = in_buf->require_frame_desc<kotekan::N2FrameDesc>()->get_support_mode()
                             == kotekan::N2SupportMode::PerProductV1;
    const double bin_width = 360.0 / num_bins_per_rotation;
    const int64_t clock_zero_ns = tel.to_time_ns(0);
    const int64_t tick_ns = tel.to_time_ns(1) - clock_zero_ns;
    if (clock_zero_ns < 0 || tick_ns <= 0)
        FATAL_ERROR("N2TimeDownsample requires a positive integer-nanosecond FPGA clock");
    const uint64_t max_tick =
        (std::numeric_limits<int64_t>::max() - static_cast<uint64_t>(clock_zero_ns)) / tick_ns;
    const EOP startup = tel.get_EOP_at_time_ns(clock_zero_ns);
    int64_t startup_rotation;
    get_ERA_from_UT1(startup.t_ut1_ns, &startup_rotation);
    const uint32_t startup_bin = static_cast<uint32_t>(startup.ERA_deg / bin_width);

    auto& skipped = Metrics::instance().add_counter("kotekan_timedownsample_skipped_frame_total",
                                                    unique_name, {"freq_id", "reason"});
    bool have_previous = false, active = false, aligned = false, have_supported = false;
    uint64_t previous_end = 0, previous_input_idx = 0, previous_bin = 0, active_bin = 0;
    int64_t previous_reference_ns = 0;
    uint32_t freq_id = 0;
    double freq_MHz = 0;
    EOP target = eop_null;
    std::vector<std::complex<float>> phase(num_elements, 1.0f);
    std::vector<std::complex<double>> vis_sum, evec_sum;
    std::vector<double> variance_sum, eval_sum;
    std::vector<uint8_t> precision_available;
    std::vector<uint64_t> product_count_sum;
    double erms_sum = 0;

    const auto finite_complex = [](const auto& z) {
        return std::isfinite(z.real()) && std::isfinite(z.imag());
    };
    const auto add_count = [&](uint64_t& total, uint64_t amount) {
        if (amount > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - total)
            FATAL_ERROR("N2TimeDownsample accumulated count overflow");
        total += amount;
    };
    const auto to_finite_float = [&](double value) {
        const float result = static_cast<float>(value);
        if (!std::isfinite(result))
            FATAL_ERROR("N2TimeDownsample supported output is not representable as finite float");
        return result;
    };
    const auto checked_eop = [&](const EOP& supplied, const EOP& expected) {
        const double delta = std::abs(supplied.ERA_deg - expected.ERA_deg);
        if (!std::isfinite(supplied.ERA_deg) || supplied.ERA_deg < 0 || supplied.ERA_deg >= 360
            || !std::isfinite(supplied.delta_UT1_inst) || !std::isfinite(supplied.xp_as)
            || !std::isfinite(supplied.yp_as)
            || std::abs(static_cast<long double>(supplied.t_inst_ns) - expected.t_inst_ns) > 5
            || std::abs(static_cast<long double>(supplied.t_ut1_ns) - expected.t_ut1_ns) > 5
            || std::min(delta, 360.0 - delta) > 1.0e-7
            || std::abs(supplied.delta_UT1_inst - expected.delta_UT1_inst) > 1.0e-12
            || std::abs(supplied.xp_as - expected.xp_as) > 1.0e-12
            || std::abs(supplied.yp_as - expected.yp_as) > 1.0e-12)
            FATAL_ERROR("N2TimeDownsample EOP/time identity mismatch");
    };
    const auto finish = [&]() {
        N2FrameView output(out_buf, output_frame_id, true);
        const auto end = output.fpga_start_tick + output.frame_length_fpga_ticks;
        const double age = 1.0e-9 * (tel.to_time_ns(end) - output.frame_start_time_ns);
        if (age > max_age) {
            skipped.labels({std::to_string(freq_id), "age"}).inc();
            active = false;
            return;
        }
        const double scalar_total = static_cast<double>(output.n_valid_fpga_ticks);
        for (size_t p = 0; p < nprod; ++p) {
            const double total =
                per_product ? static_cast<double>(product_count_sum[p]) : scalar_total;
            if (per_product)
                output.valid_fpga_ticks[p] = product_count_sum[p];
            if (total > 0) {
                const auto mean = vis_sum[p] / total;
                output.vis[p] = {to_finite_float(mean.real()), to_finite_float(mean.imag())};
                const double precision = precision_available[p] && variance_sum[p] > 0
                                             ? total * total / variance_sum[p]
                                             : 0.0;
                const float weight = static_cast<float>(precision);
                output.weight[p] = std::isfinite(weight) && weight > 0 ? weight : 0.0f;
            } else {
                output.vis[p] = 0.0f;
                output.weight[p] = 0.0f;
            }
        }
        // A heterogeneous set of baseline means has no common sample count
        // for eigen diagnostics. Mark these unavailable instead of inventing one.
        const double total = per_product ? 0.0 : scalar_total;
        if (per_product) {
            output.emethod = kotekan::N2EigenMethod::none;
            std::fill(output.radiometer_chi2.begin(), output.radiometer_chi2.end(), -1.0f);
        }
        for (size_t k = 0; k < eval_sum.size(); ++k)
            output.eval[k] = total > 0 ? to_finite_float(eval_sum[k] / total) : 0.0f;
        for (size_t k = 0; k < evec_sum.size(); ++k) {
            const auto mean = total > 0 ? evec_sum[k] / total : std::complex<double>{};
            output.evec[k] = {to_finite_float(mean.real()), to_finite_float(mean.imag())};
        }
        output.erms = total > 0 ? to_finite_float(erms_sum / total) : (per_product ? -1.0f : 0.0f);
        if (!have_supported) {
            std::fill(output.flags.begin(), output.flags.end(), 0.0f);
            std::fill(output.mask.begin(), output.mask.end(), 0u);
            std::fill(output.gain.begin(), output.gain.end(), N2::cfloat{-1.0f, 0.0f});
            std::fill(output.radiometer_chi2.begin(), output.radiometer_chi2.end(), -1.0f);
        }
        output.time_center_eop = tel.get_EOP_at_time_ns(
            tel.to_time_ns(output.fpga_start_tick + output.frame_length_fpga_ticks / 2));
        out_buf->mark_frame_full(unique_name, output_frame_id++);
        active = false;
    };

    while (!stop_thread) {
        if (in_buf->wait_for_full_frame(unique_name, frame_id) == nullptr)
            break;
        N2FrameView frame(in_buf, frame_id, true);
        const uint64_t seq = frame.fpga_start_tick, length = frame.frame_length_fpga_ticks;
        const uint64_t count = frame.n_valid_fpga_ticks, rfi = frame.n_rfi_fpga_ticks;
        const uint64_t rfi_only = frame.n_rfi_only_fpga_ticks, lost = frame.n_pl_fpga_ticks;
        if (frame.n2_layout != N2Layout::FullUpperTri || frame.num_elements != num_elements
            || frame.num_prod != num_elements * (num_elements + 1) / 2)
            FATAL_ERROR(
                "N2TimeDownsample requires a full upper triangle with the configured inputs");
        if (length == 0 || seq > max_tick || length > max_tick - seq)
            FATAL_ERROR("N2TimeDownsample invalid or overflowing FPGA time interval");
        const uint64_t end = seq + length;
        if (per_product) {
            if (count || rfi || rfi_only || lost)
                FATAL_ERROR("N2TimeDownsample per_product_v1 scalar support counters must be "
                            "unavailable (zero)");
            for (const uint64_t n : frame.valid_fpga_ticks)
                if (n > length)
                    FATAL_ERROR("N2TimeDownsample per-product valid count exceeds frame interval");
        } else if (count > length || lost > length - count || rfi_only != length - count - lost
                   || rfi > length || rfi < rfi_only || rfi - rfi_only > lost) {
            FATAL_ERROR("N2TimeDownsample inconsistent valid/RFI/packet-loss counts");
        }
        const bool any_supported =
            per_product ? std::any_of(frame.valid_fpga_ticks.begin(), frame.valid_fpga_ticks.end(),
                                      [](uint64_t n) { return n > 0; })
                        : count > 0;
        if (frame.frame_start_time_ns != static_cast<uint64_t>(tel.to_time_ns(seq)))
            FATAL_ERROR("N2TimeDownsample FPGA/start-time identity mismatch");
        if (!std::isfinite(frame.freq_MHz) || frame.freq_MHz < 0)
            FATAL_ERROR("N2TimeDownsample invalid physical frequency");
        if (frame.bin_eop.t_inst_ns < clock_zero_ns)
            FATAL_ERROR("N2TimeDownsample invalid reference time");
        const EOP input_eop = tel.get_EOP_at_time_ns(frame.bin_eop.t_inst_ns);
        checked_eop(frame.bin_eop, input_eop);
        checked_eop(frame.time_center_eop,
                    tel.get_EOP_at_time_ns(tel.to_time_ns(seq + length / 2)));
        int64_t rotation;
        get_ERA_from_UT1(input_eop.t_ut1_ns, &rotation);
        const uint32_t local_bin = static_cast<uint32_t>(input_eop.ERA_deg / bin_width);
        const __int128 bin_wide =
            static_cast<__int128>(rotation - startup_rotation) * num_bins_per_rotation + local_bin
            - startup_bin;
        if (bin_wide < 0 || bin_wide > std::numeric_limits<uint64_t>::max())
            FATAL_ERROR("N2TimeDownsample absolute bin index overflow");
        const uint64_t bin = static_cast<uint64_t>(bin_wide);
        if (!have_previous) {
            freq_id = frame.freq_id;
            freq_MHz = frame.freq_MHz;
            nprod = frame.num_prod;
            num_eigenvectors = frame.num_ev;
            previous_bin = bin;
            vis_sum.resize(nprod);
            variance_sum.resize(nprod);
            precision_available.resize(nprod);
            product_count_sum.resize(nprod);
            eval_sum.resize(num_eigenvectors);
            evec_sum.resize(num_eigenvectors * num_elements);
        } else {
            if (frame.freq_id != freq_id || frame.freq_MHz != freq_MHz
                || frame.num_ev != num_eigenvectors)
                FATAL_ERROR("N2TimeDownsample frequency/layout identity changed");
            if (seq != previous_end || previous_input_idx == std::numeric_limits<uint64_t>::max()
                || frame.abs_time_idx != previous_input_idx + 1
                || input_eop.t_inst_ns <= previous_reference_ns || bin < previous_bin)
                FATAL_ERROR("N2TimeDownsample noncontiguous or out-of-order input identity");
            if (bin != previous_bin)
                aligned = true;
        }
        previous_end = end;
        previous_input_idx = frame.abs_time_idx;
        previous_reference_ns = input_eop.t_inst_ns;
        previous_bin = bin;
        have_previous = true;
        for (size_t p = 0; p < nprod; ++p)
            if ((per_product ? frame.valid_fpga_ticks[p] : count) > 0
                && !finite_complex(frame.vis[p]))
                FATAL_ERROR("N2TimeDownsample nonfinite supported visibility");
        if (!aligned) {
            skipped.labels({std::to_string(freq_id), "alignment"}).inc();
            in_buf->mark_frame_empty(unique_name, frame_id++);
            continue;
        }
        if (active && bin != active_bin)
            finish();
        if (!active) {
            if (out_buf->wait_for_empty_frame(unique_name, output_frame_id) == nullptr)
                break;
            auto output = N2FrameView::copy_frame(in_buf, frame_id, out_buf, output_frame_id, true);
            active_bin = bin;
            active = true;
            have_supported = false;
            target = tel.get_EOP_at_UT1(get_UT1_from_ERA(rotation, (local_bin + 0.5) * bin_width));
            output.bin_eop = target;
            output.bin_start_ERA_deg = local_bin * bin_width;
            output.bin_end_ERA_deg = (local_bin + 1) * bin_width;
            auto start_eop = tel.get_EOP_at_UT1(get_UT1_from_ERA(rotation, local_bin * bin_width));
            auto end_eop =
                tel.get_EOP_at_UT1(get_UT1_from_ERA(rotation, (local_bin + 1) * bin_width));
            output.bin_start_ERAL_deg = tel.get_ERAL_deg(start_eop);
            output.bin_end_ERAL_deg = tel.get_ERAL_deg(end_eop);
            output.abs_time_idx = bin;
            output.frame_length_fpga_ticks = output.n_valid_fpga_ticks = 0;
            output.n_rfi_fpga_ticks = output.n_rfi_only_fpga_ticks = output.n_pl_fpga_ticks = 0;
            std::fill(vis_sum.begin(), vis_sum.end(), std::complex<double>{});
            std::fill(variance_sum.begin(), variance_sum.end(), 0.0);
            std::fill(precision_available.begin(), precision_available.end(), 1u);
            std::fill(product_count_sum.begin(), product_count_sum.end(), 0u);
            std::fill(eval_sum.begin(), eval_sum.end(), 0.0);
            std::fill(evec_sum.begin(), evec_sum.end(), std::complex<double>{});
            erms_sum = 0;
        }
        N2FrameView output(out_buf, output_frame_id, true);
        if (frame.dataset_id != output.dataset_id
            || frame.rfi_frame_excision_enabled != output.rfi_frame_excision_enabled
            || frame.rfi_frame_excision_num != output.rfi_frame_excision_num
            || frame.rfi_frame_excision_threshold != output.rfi_frame_excision_threshold
            || frame.rfi_frame_excision_fraction != output.rfi_frame_excision_fraction)
            FATAL_ERROR("N2TimeDownsample dataset or RFI policy changed within an output bin");
        add_count(output.frame_length_fpga_ticks, length);
        add_count(output.n_valid_fpga_ticks, count);
        add_count(output.n_rfi_fpga_ticks, rfi);
        add_count(output.n_rfi_only_fpga_ticks, rfi_only);
        add_count(output.n_pl_fpga_ticks, lost);
        if (any_supported) {
            if (!have_supported) {
                std::copy(frame.flags.begin(), frame.flags.end(), output.flags.begin());
                std::copy(frame.mask.begin(), frame.mask.end(), output.mask.begin());
                std::copy(frame.gain.begin(), frame.gain.end(), output.gain.begin());
                std::copy(frame.radiometer_chi2.begin(), frame.radiometer_chi2.end(),
                          output.radiometer_chi2.begin());
                output.emethod = frame.emethod;
                have_supported = true;
            } else if (!std::equal(frame.mask.begin(), frame.mask.end(), output.mask.begin())
                       || !std::equal(frame.gain.begin(), frame.gain.end(), output.gain.begin())
                       || frame.emethod != output.emethod) {
                FATAL_ERROR("N2TimeDownsample input mask/gains/eigenmethod changed within a bin");
            }
            // A feed flagged in any contributing frame remains flagged.
            for (size_t i = 0; i < num_elements; ++i)
                if (frame.flags[i] == 0.0f)
                    output.flags[i] = 0.0f;
            if (do_fringestop)
                tel.fill_fringestop_phases_1d(freq_MHz, input_eop, target, feed_positions_m, phase);
            size_t p = 0;
            for (size_t i = 0; i < num_elements; ++i) {
                for (size_t j = i; j < num_elements; ++j, ++p) {
                    const uint64_t n_ticks = per_product ? frame.valid_fpga_ticks[p] : count;
                    if (!n_ticks)
                        continue;
                    if (per_product)
                        add_count(product_count_sum[p], n_ticks);
                    const double n = static_cast<double>(n_ticks);
                    const std::complex<double> multiplier =
                        std::complex<double>(phase[i]) * std::conj(std::complex<double>(phase[j]));
                    if (!finite_complex(multiplier))
                        FATAL_ERROR("N2TimeDownsample nonfinite fringe phase");
                    vis_sum[p] += n * multiplier * std::complex<double>(frame.vis[p]);
                    const double weight = frame.weight[p];
                    if (!(weight > 0) || !std::isfinite(weight))
                        precision_available[p] = 0;
                    else
                        variance_sum[p] += n * n * std::norm(multiplier) / weight;
                }
            }
            const double n = static_cast<double>(count);
            if (!per_product) {
                for (size_t k = 0; k < num_eigenvectors; ++k) {
                    if (!std::isfinite(frame.eval[k]))
                        FATAL_ERROR("N2TimeDownsample nonfinite supported eigenvalue");
                    eval_sum[k] += n * frame.eval[k];
                    for (size_t j = 0; j < num_elements; ++j) {
                        const size_t index = k * num_elements + j;
                        if (!finite_complex(frame.evec[index]))
                            FATAL_ERROR("N2TimeDownsample nonfinite supported eigenvector");
                        const std::complex<double> multiplier =
                            std::complex<double>(phase[j])
                            * std::conj(std::complex<double>(phase[0]));
                        evec_sum[index] += n * multiplier * std::complex<double>(frame.evec[index]);
                    }
                }
                if (!std::isfinite(frame.erms))
                    FATAL_ERROR("N2TimeDownsample nonfinite supported eigen residual");
                erms_sum += n * frame.erms;
            }
        }
        in_buf->mark_frame_empty(unique_name, frame_id++);
    }
    // The first and final partial ERA bins are intentionally not published.
}
