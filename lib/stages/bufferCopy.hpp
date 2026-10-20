#ifndef BUFFER_COPY_HPP
#define BUFFER_COPY_HPP

#include "Config.hpp" // for Config
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Gauge
#include "visUtil.hpp"

#include <string> // for string
#include <tuple>  // for tuple
#include <vector> // for vector

/**
 * @brief Copies frames from one buffer into many buffers.
 *
 *
 * @warning The sizes of the frames must be the same in all buffers, and the
 *       metadata types and underlying pools must also be the same.
 *
 * With @c detach_timeout_s above zero the stage waits at most that long for an empty
 * frame in an output buffer; then that output is detached (logged and counted) and
 * gets no more frames for the rest of the run, so a stalled consumer of one copy
 * cannot hold the input buffer. Default 0: wait for every output without a limit.
 *
 * @conf detach_timeout_s Float. Seconds to wait for a full output before detaching it
 *       (default 0: never detach).
 *
 * @par Metrics
 * @metric kotekan_buffercopy_detached_outputs Number of detached output buffers.
 *
 *
 * @par Buffer
 * @buffer in_buf Buffer to copy frames from
 *        @buffer_format Matches the output buffers
 *        @buffer_metadata Matches the output buffers
 * @buffer out_bufs Array of output buffers to copy frames into.
 *                 This is a named array of output buffers in the format:
 *                 - buffer_name_0
 *                 - buffer_name_1
 *                 .
 *        @buffer_format any, but all must be the same type.
 *        @buffer_metadata any, but all must be the same type.
 *
 * @author James Willis
 */
class bufferCopy : public kotekan::Stage {
public:
    /// Constructor
    bufferCopy(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~bufferCopy() = default;

    /// Thread for merging the frames.
    void main_thread() override;

protected:
    /// Wait up to detach_timeout_s for frame_id of out_buf to be empty.
    bool wait_for_empty(Buffer* out_buf, int frame_id);

    /// The input buffer to copy frames from
    Buffer* in_buf;

    /// Config variables
    /// Flag to copy metadata or not
    bool _copy_metadata;
    /// Seconds to wait for a full output before detaching it; 0 never detaches
    double _detach_timeout_s;
    kotekan::prometheus::Gauge* detached_gauge = nullptr;

    /// Array of output buffers to copy frames into
    /// Items are "internal_name", "buffer", "frame_id", "use_memcpy"
    std::vector<std::tuple<std::string, Buffer*, frameID>> out_bufs;
};

#endif
