#include "bufferCopy.hpp"

#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "kotekanLogging.hpp"    // for INFO, DEBUG2, ERROR, FATAL_ERROR
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "visUtil.hpp"           // for frameID, modulo

#include "fmt.hpp"  // for compile_string_to_view, format, fmt
#include "json.hpp" // for json, basic_json, iter_impl

#include <algorithm>  // for count
#include <chrono>     // for steady_clock, duration
#include <cstring>    // for memcpy
#include <functional> // for bind, function
#include <memory>     // for shared_ptr
#include <stdexcept>  // for runtime_error, invalid_argument
#include <stdint.h>   // for uint8_t
#include <thread>     // for sleep_for


using nlohmann::json;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferCopy);

bufferCopy::bufferCopy(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferCopy::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    _copy_metadata = config.get_default<bool>(unique_name, "copy_metadata", false);
    _detach_timeout_s = config.get_default<double>(unique_name, "detach_timeout_s", 0.0);
    if (!(_detach_timeout_s >= 0.0))
        throw std::invalid_argument("bufferCopy: detach_timeout_s must be nonnegative");
    if (_detach_timeout_s > 0.0) {
        detached_gauge = &kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_buffercopy_detached_outputs", unique_name);
        detached_gauge->set(0);
    }

    json buffer_list = config.get<std::vector<json>>(unique_name, "out_bufs");
    Buffer* out_buf = nullptr;
    std::string internal_name;
    std::string buffer_name;
    INFO("buffer_list {:s}", buffer_list.dump());
    for (json buffer : buffer_list) {

        if (buffer.is_object()) {
            // Assuming each array entry has only one key.
            json::iterator it = buffer.begin();
            internal_name = it.key();
            buffer_name = it.value().get<std::string>();
            out_buf = buffer_container.get_buffer(buffer_name);
            if (out_buf == nullptr)
                throw std::runtime_error(
                    fmt::format(fmt("Output buffer is null: {:s}"), buffer.dump()));
        } else if (buffer.is_string()) {
            buffer_name = buffer.get<std::string>();
            internal_name = "";
            out_buf = buffer_container.get_buffer(buffer_name);
            if (out_buf == nullptr)
                throw std::runtime_error(
                    fmt::format(fmt("Output buffer is null: {:s}"), buffer.dump()));
        } else {
            throw std::runtime_error(
                fmt::format(fmt("Unknown value in out_bufs: {:s}"), buffer.dump()));
        }

        if (out_buf->frame_size != in_buf->frame_size) {
            throw std::invalid_argument(fmt::format(fmt("Input buffer '{:s}' not equal to output "
                                                        "buffer size."),
                                                    buffer_name));
        }

        out_buf->register_producer(unique_name);
        INFO("Adding buffer: {:s}:{:s}", internal_name, out_buf->buffer_name);
        out_bufs.push_back(std::make_tuple(internal_name, out_buf, frameID(out_buf)));
    }
}

bool bufferCopy::wait_for_empty(Buffer* out_buf, const int frame_id) {
    // This stage is the only producer, so a frame found empty stays empty.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(_detach_timeout_s);
    while (!out_buf->is_frame_empty(frame_id)) {
        if (stop_thread || std::chrono::steady_clock::now() >= deadline)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

void bufferCopy::main_thread() {

    frameID in_frame_id(in_buf);
    std::vector<bool> detached(out_bufs.size(), false);
    int64_t frames_copied = 0;

    while (!stop_thread) {
        uint8_t* input_frame = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (input_frame == nullptr)
            break;

        for (std::size_t i = 0; i < out_bufs.size(); ++i) {
            auto& buffer_info = out_bufs[i];
            if (detached[i])
                continue;

            const std::string& internal_buffer_name = std::get<0>(buffer_info);
            Buffer* out_buf = std::get<1>(buffer_info);
            frameID& out_frame_id = std::get<2>(buffer_info);

            if (out_buf->get_num_producers() != 1) {
                FATAL_ERROR("Cannot copy into buffer: {:s} as it has more than one producer.",
                            internal_buffer_name);
                return;
            }

            if (_detach_timeout_s > 0.0 && !wait_for_empty(out_buf, out_frame_id)) {
                if (stop_thread)
                    break;
                detached[i] = true;
                detached_gauge->set(std::count(detached.begin(), detached.end(), true));
                ERROR("Output buffer {:s} stayed full for {:.3f} s after {:d} input frames: its "
                      "consumer has fallen behind, so it is DETACHED and gets no more frames "
                      "until a restart",
                      out_buf->buffer_name, _detach_timeout_s, frames_copied);
                continue;
            }

            /// Wait for an output frame
            DEBUG2("Waiting for {:s}[{:d}]", out_buf->buffer_name, out_frame_id);
            uint8_t* output_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
            if (output_frame == nullptr)
                goto exit_loop; // Shutdown condition

            // Either make a deep copy or pass the metadata depending if the flag is set
            if (in_buf->get_metadata(in_frame_id)) {
                if (_copy_metadata) {
                    out_buf->allocate_new_metadata_object(out_frame_id);
                    in_buf->copy_metadata(in_frame_id, out_buf, out_frame_id);
                } else
                    in_buf->pass_metadata(in_frame_id, out_buf, out_frame_id);
            }

            // Copy the frame.
            std::memcpy(output_frame, input_frame, in_buf->frame_size);

            out_buf->mark_frame_full(unique_name, out_frame_id);
            out_frame_id++;
        }

        // We always release the input buffer even if it isn't selected.
        in_buf->mark_frame_empty(unique_name, in_frame_id);

        // Increase the in_frame_id for the input buffer
        in_frame_id++;
        frames_copied++;
    }
exit_loop:;
}
