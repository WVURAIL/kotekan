#ifndef N2_TIME_DOWNSAMP_HPP
#define N2_TIME_DOWNSAMP_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "geoUtil.hpp"

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

/**
 * @class N2TimeDownsample
 * @brief Count-weighted downsampling of a continuous, single-frequency N2 stream.
 *
 * For each product, V = sum(N_i V_i) / sum(N_i). With independent input-frame
 * errors and supplied inverse variances w_i, the output inverse variance is
 * (sum N_i)^2 / sum(N_i^2 / w_i). Fringe phases enter the coefficients.
 * A positive-count input with zero, negative or nonfinite precision makes that
 * product's output precision unavailable (zero), while its finite mean remains
 * supported. Zero-count payload is unused, including nonfinite values.
 * Nonfinite supported visibility/eigen data is rejected.
 *
 * Matching FullUpperTri N2 buffers are required. FPGA intervals and input
 * indices must be continuous; frequency, telescope time and EOP identities are
 * checked. Dataset/RFI policy and supported flags/gains/eigenmethod may not
 * change inside a bin. Bins include the Earth-rotation number. Initial and final
 * partial bins are discarded. time_center_eop describes the actual interval
 * midpoint; bin_eop and local ERA edges describe the nominal output bin.
 *
 * Scalar counts represent common support. With support_mode=per_product_v1,
 * valid_fpga_ticks supplies counts separately by product and scalar count/loss
 * fields are unavailable (zero). The output preserves authoritative per-product
 * counts. Cross-frame covariance is not available and is not inferred.
 * Scalar eigen diagnostics retain the count-weighted convention; per-product
 * eigen/radiometer diagnostics are marked unavailable rather than inventing a
 * common count. Other diagnostics come from the first supported frame.
 *
 * @buffer in_buf FullUpperTri N2 frames with N2Metadata.
 * @buffer out_buf Matching FullUpperTri N2 frames with N2Metadata.
 * @conf num_bins_per_rotation Positive number of output bins per rotation.
 * @conf max_age Positive finite maximum accumulated span in seconds (default 120).
 * @conf do_fringestop Apply phases to the output bin center (default true).
 * @conf num_elements Number of inputs, matching the descriptor.
 */
class N2TimeDownsample : public kotekan::Stage {
public:
    N2TimeDownsample(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    void main_thread() override;

private:
    size_t num_elements, num_eigenvectors, nprod;
    uint32_t num_bins_per_rotation;
    float max_age;
    bool do_fringestop;
    std::vector<vec3d_t> feed_positions_m;
    Buffer* in_buf;
    Buffer* out_buf;
};

#endif
