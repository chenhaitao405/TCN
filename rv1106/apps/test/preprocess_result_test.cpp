// Compare preprocessing results between:
// 1) NCHW_float32_to_NC1HWC2_int8
// 2) fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "boost/circular_buffer.hpp"
#include "data_stream.h"
#include "model_process/model_info_parse.hpp"
#include "model_process/model_setting.h"

namespace {

constexpr int kChannel = Hip::ModelChannel;

void fill_random_buffer(boost::circular_buffer<Hip::DataFrame>& buffer,
						int total_frames,
						uint32_t seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

	for (int i = 0; i < total_frames; ++i) {
		Hip::DataFrame frame{};
		for (int c = 0; c < kChannel; ++c) {
			frame.channels[c] = dist(rng);
		}
		buffer.push_back(frame);
	}
}

std::vector<float> circular_buffer_to_nchw(const boost::circular_buffer<Hip::DataFrame>& buffer,
										   int w) {
	std::vector<float> nchw(kChannel * w, 0.0f);
	const int offset = static_cast<int>(buffer.size()) - w;

	// N=1, H=1, so contiguous layout is [C][W]
	for (int c = 0; c < kChannel; ++c) {
		for (int x = 0; x < w; ++x) {
			nchw[c * w + x] = buffer[offset + x].channels[c];
		}
	}
	return nchw;
}

bool run_one_case(int total_frames,
				  int w,
				  float scale,
				  int zp,
				  const float* mean,
				  const float* std,
				  int mean_std_len,
				  uint32_t seed,
				  int case_id) {
	if (total_frames < w) {
		std::cerr << "[ERROR] total_frames < w in case " << case_id << '\n';
		return false;
	}

	boost::circular_buffer<Hip::DataFrame> buffer(total_frames);
	fill_random_buffer(buffer, total_frames, seed);

	std::vector<int8_t> out_fast(w * ALIGNED_CHANNEL, 0);
	std::vector<int8_t> out_nchw(w * ALIGNED_CHANNEL, 0);

	const int ret_fast = fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8(
		buffer,
		out_fast.data(),
		w,
		scale,
		zp,
		mean,
		std,
		mean_std_len);

	if (ret_fast != 0) {
		std::cerr << "[ERROR] fast preprocess failed in case " << case_id
				  << ", ret=" << ret_fast << '\n';
		return false;
	}

	std::vector<float> nchw = circular_buffer_to_nchw(buffer, w);
	const int ret_nchw = NCHW_float32_to_NC1HWC2_int8(
		nchw.data(),
		out_nchw.data(),
		1,
		kChannel,
		1,
		w,
		scale,
		zp,
		mean,
		std,
		mean_std_len);

	if (ret_nchw != 0) {
		std::cerr << "[ERROR] NCHW preprocess failed in case " << case_id
				  << ", ret=" << ret_nchw << '\n';
		return false;
	}

	int mismatch_count = 0;
	for (int i = 0; i < w * ALIGNED_CHANNEL; ++i) {
		if (out_fast[i] != out_nchw[i]) {
			if (mismatch_count < 16) {
				const int x = i / ALIGNED_CHANNEL;
				const int c = i % ALIGNED_CHANNEL;
				std::cerr << "[Mismatch] case=" << case_id
						  << " x=" << x
						  << " c=" << c
						  << " fast=" << static_cast<int>(out_fast[i])
						  << " nchw=" << static_cast<int>(out_nchw[i])
						  << '\n';
			}
			++mismatch_count;
		}
	}

	if (mismatch_count > 0) {
		std::cerr << "[FAIL] case " << case_id
				  << " mismatch bytes=" << mismatch_count
				  << " / " << (w * ALIGNED_CHANNEL) << '\n';
		return false;
	}

	std::cout << "[PASS] case " << std::setw(2) << case_id
			  << " total_frames=" << total_frames
			  << " w=" << w
			  << " scale=" << scale
			  << " zp=" << zp
			  << " seed=" << seed
			  << '\n';
	return true;
}

} // namespace

int main() {
	std::cout << "Compare NCHW_float32_to_NC1HWC2_int8 vs "
				 "fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8\n";

	const struct TestCfg {
		int total_frames;
		int w;
		float scale;
		int zp;
		uint32_t seed;
	} cases[] = {
		{128, 128, 0.06125f, -128, 1u},
		{256, 128, 0.06125f, -128, 2u},
		{200, 128, 0.03750f, -3, 3u},
		{129, 128, 0.12500f, 7, 4u},
		{300, 128, 0.00800f, 12, 5u},
	};

	int pass_count = 0;
	const int total_cases = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
	for (int i = 0; i < total_cases; ++i) {
		const bool ok = run_one_case(
			cases[i].total_frames,
			cases[i].w,
			cases[i].scale,
			cases[i].zp,
			Hip::g_mean_vals,
			Hip::g_std_vals,
			Hip::ModelChannel,
			cases[i].seed,
			i + 1);
		if (!ok) {
			std::cerr << "\nResult: FAIL\n";
			return 1;
		}
		++pass_count;
	}

	std::cout << "\nResult: PASS (" << pass_count << "/" << total_cases << ")\n";
	return 0;
}
