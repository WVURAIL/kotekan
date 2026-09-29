#define BOOST_TEST_MODULE "test_HFBAccumulate"

#include "CHORDTelescope.hpp"
#include "Config.hpp"
#include "HFBAccumulate.hpp"
#include "HFBFrameView.hpp"
#include "HFBMetadata.hpp"
#include "Telescope.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "configUpdater.hpp"
#include "datasetManager.hpp"
#include "metadata.hpp"
#include "test_utils.hpp"
#include "visUtil.hpp"

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

struct HFBEnvironment {
    HFBEnvironment() {
        nlohmann::json cfg = {{"num_polarizations", 2},
                              {"dataset_manager", {{"use_dataset_broker", false}}}};
        add_test_telescope_config(cfg);
        kotekan::Config config;
        config.update_config(cfg);
        kotekan::configUpdater::instance().apply_config(config);
        if (!FACTORY(Telescope)::exists("CHORDTelescope"))
            FACTORY(Telescope)::register_type<CHORDTelescope>("CHORDTelescope");
        Telescope::instance(config);
        datasetManager::instance(config);
    }
};

BOOST_TEST_GLOBAL_FIXTURE(HFBEnvironment);

struct HFBBuffers {
    static constexpr int depth = 6;
    std::shared_ptr<metadataPool> input_pool =
        metadataPool::create(2 * depth, sizeof(chordMetadata), "hfb_input_pool", "chordMetadata");
    std::shared_ptr<metadataPool> output_pool =
        metadataPool::create(2, sizeof(HFBMetadata), "hfb_output_pool", "HFBMetadata");
    Buffer input{depth, sizeof(float), input_pool, "hfb_input",        "standard",
                 0,     false,         false,      std::vector<int>{}, true};
    Buffer lost{depth, sizeof(uint32_t),   input_pool, "hfb_lost", "standard", 0, false,
                false, std::vector<int>{}, true};
    Buffer output{2,
                  HFBFrameView::calculate_frame_size(1, 1),
                  output_pool,
                  "hfb_output",
                  "hfb",
                  0,
                  false,
                  false,
                  std::vector<int>{},
                  true};
    kotekan::bufferContainer buffers;

    HFBBuffers() {
        buffers.add_buffer("hfb_input", &input);
        buffers.add_buffer("hfb_lost", &lost);
        buffers.add_buffer("hfb_output", &output);
        input.register_producer("test_producer");
        lost.register_producer("test_producer");
        output.register_consumer("test_consumer");
    }

    kotekan::Config config(uint32_t samples, uint32_t frames) {
        kotekan::Config result;
        result.update_config({{"stage",
                               {{"cpu_affinity", nlohmann::json::array()},
                                {"log_level", "WARN"},
                                {"join_timeout", 5},
                                {"hfb_input_buf", "hfb_input"},
                                {"compressed_lost_samples_buf", "hfb_lost"},
                                {"hfb_output_buf", "hfb_output"},
                                {"samples_per_data_set", samples},
                                {"num_frames_to_integrate", frames},
                                {"num_frb_total_beams", 1},
                                {"factor_upchan", 1},
                                {"freq_ids", {0}},
                                {"good_samples_threshold", 0.9}}}});
        return result;
    }

    void stop(HFBAccumulate& stage) {
        stage.stop();
        input.send_shutdown_signal();
        lost.send_shutdown_signal();
        output.send_shutdown_signal();
        stage.join();
    }
};

BOOST_AUTO_TEST_CASE(integration_metadata_exceeds_32_bit_sample_count) {
    constexpr uint32_t frames = 5;
    for (uint32_t samples : {1U << 30, 3U << 30}) {
        HFBBuffers fixture;
        auto config = fixture.config(samples, frames);
        HFBAccumulate stage(config, "/stage", fixture.buffers);

        // Only one beam value per frame is stored; the large count lives in metadata.
        for (int frame = 0; frame < HFBBuffers::depth; ++frame) {
            auto* data = fixture.input.wait_for_empty_frame("test_producer", frame);
            *reinterpret_cast<float*>(data) = static_cast<float>(samples) * (1 + frame % 2);
            fixture.input.allocate_new_metadata_object(frame);
            auto meta = get_chord_metadata(&fixture.input, frame);
            meta->set_fpga_seq_num(static_cast<int64_t>(frame) * samples);
            meta->set_coarse_freq({0});
            meta->set_dataset_id(dset_id_t::null);
            fixture.input.mark_frame_full("test_producer", frame);

            auto* lost_data = fixture.lost.wait_for_empty_frame("test_producer", frame);
            *reinterpret_cast<uint32_t*>(lost_data) = 0;
            fixture.lost.allocate_new_metadata_object(frame);
            auto lost_meta = get_chord_metadata(&fixture.lost, frame);
            lost_meta->set_fpga_seq_num(static_cast<int64_t>(frame) * samples);
            lost_meta->set_lost_timesamples(0);
            fixture.lost.mark_frame_full("test_producer", frame);
        }

        stage.start();
        const auto timeout = double_to_ts(current_time() + 5);
        const int status = fixture.output.wait_for_full_frame_timeout("test_consumer", 0, timeout);
        if (status == 0) {
            HFBFrameView output(&fixture.output, 0);
            BOOST_CHECK_EQUAL(output.fpga_seq_start, 0);
            BOOST_CHECK_EQUAL(output.fpga_seq_total, uint64_t{samples} * frames);
            BOOST_CHECK_EQUAL(output.fpga_seq_length, uint64_t{samples} * frames);
            BOOST_CHECK_CLOSE(output.hfb[0], 1.4f, 0.001f);
            BOOST_CHECK(std::isfinite(output.weight[0]));
            BOOST_CHECK_GT(output.weight[0], 0.0f);
            BOOST_CHECK_CLOSE(output.weight[0], 25.0f / 3.0f, 0.001f);
            fixture.output.mark_frame_empty("test_consumer", 0);
        }
        fixture.stop(stage);
        BOOST_REQUIRE_EQUAL(status, 0);
    }
}

BOOST_AUTO_TEST_CASE(reject_invalid_integration_durations) {
    for (const auto& [samples, frames] : std::vector<std::pair<uint32_t, uint32_t>>{
             {0, 5},
             {1, 0},
             {std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()}}) {
        HFBBuffers fixture;
        auto config = fixture.config(samples, frames);
        BOOST_CHECK_THROW(HFBAccumulate(config, "/stage", fixture.buffers), std::invalid_argument);
    }
}


BOOST_AUTO_TEST_CASE(hfb_layout_uses_wide_sample_counts) {
    BOOST_CHECK_EQUAL(HFBFrameView::calculate_frame_size(65536, 65536), 34359738368ULL);
    BOOST_CHECK_THROW(HFBFrameView::calculate_buffer_layout(std::numeric_limits<uint32_t>::max(),
                                                            std::numeric_limits<uint32_t>::max()),
                      std::overflow_error);
}
