#include "rknn/rknn_api.h"
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "model_process/model_process.h"
#include "npy/npy.h"
#include "fp16/Float16.h"
#include "boost/circular_buffer.hpp"

void send_int8_list_adb(const int8_t* data, size_t size) {
	FILE* tmp = fopen("/tmp/adb_data.bin", "wb");
	if (tmp) {
		fwrite(data, 1, size, tmp);
		fclose(tmp);
		printf("output saved to /tmp/adb_data.bin\n");
	} else {
		printf("failed to write /tmp/adb_data.bin\n");
	}
}

static int load_npy_to_hip_circular_buffer(const NpyArray& npy,
										   boost::circular_buffer<Hip::DataFrame>& buffer,
										   int model_w) {
	// Expect NCHW float32 with shape [1, 8, 1, W]
	if (npy.dtype != "<f4" && npy.dtype != "|f4" && npy.dtype != "float32") {
		printf("unsupported npy dtype: %s (expect float32)\n", npy.dtype.c_str());
		return -1;
	}
	if (npy.shape.size() != 4) {
		printf("unsupported npy rank: %zu (expect 4D NCHW)\n", npy.shape.size());
		return -1;
	}

	const int n = static_cast<int>(npy.shape[0]);
	const int c = static_cast<int>(npy.shape[1]);
	const int h = static_cast<int>(npy.shape[2]);
	const int w = static_cast<int>(npy.shape[3]);

	if (n != 1 || c != Hip::ModelChannel || h != 1) {
		printf("unexpected npy shape=[%d,%d,%d,%d], expect [1,%d,1,W]\n", n, c, h, w, Hip::ModelChannel);
		return -1;
	}
	if (w < model_w) {
		printf("npy width too small: w=%d < model_w=%d\n", w, model_w);
		return -1;
	}

	const float* src = reinterpret_cast<const float*>(npy.data.data());
	const int start_w = w - model_w;

	for (int x = 0; x < model_w; ++x) {
		Hip::DataFrame frame{};
		for (int ch = 0; ch < Hip::ModelChannel; ++ch) {
			frame.channels[ch] = src[ch * w + (start_w + x)];
		}
		buffer.push_back(frame);
	}
	return 0;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		printf("Usage:%s model_path npy_path\n", argv[0]);
		return -1;
	}

	char* model_path = argv[1];
	char* npy_path = argv[2];
	NpyArray npy;
	if (load_npy(npy_path, npy)) {
		size_t elements = npy.data.size() / npy.elem_size;
		printf("read %s, dtype=%s, elements=%zu, shape=[", npy_path, npy.dtype.c_str(), elements);
		for (size_t i = 0; i < npy.shape.size(); ++i) {
			printf("%zu%s", npy.shape[i], (i + 1 == npy.shape.size()) ? "" : ", ");
		}
		printf("]\n");
	} else {
		return -1;
	}

	rknn_context ctx;
	std::vector<rknn_tensor_attr> input_attrs;
	std::vector<rknn_tensor_attr> output_attrs;
	int ret = model_info_parse(ctx, const_cast<char*>(model_path), input_attrs, output_attrs);
	if (ret != RKNN_SUCC) {
		printf("model info parse failed!\n");
		return -1;
	}

	rknn_tensor_type input_type = RKNN_TENSOR_INT8;
	rknn_tensor_format input_layout = RKNN_TENSOR_NC1HWC2;

	rknn_tensor_mem* input_mems[1];
	rknn_tensor_mem* output_mems[1] = {nullptr};
	float* result = nullptr;
	rknpu2::float16* out_fp16 = nullptr;
	input_attrs[0].type = input_type;
	input_attrs[0].fmt = input_layout;
	input_attrs[0].pass_through = 1;
	input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

	const int model_h = input_attrs[0].dims[2];
	const int model_w = input_attrs[0].dims[3];
	const int model_frames = model_h * model_w;

	if (model_frames != Hip::STREAM_LENGTH) {
		printf("warning: model frames=%d, Hip::STREAM_LENGTH=%d. "
			   "rf_data_collection_hip uses STREAM_LENGTH as window.\n",
			   model_frames, Hip::STREAM_LENGTH);
	}

	boost::circular_buffer<Hip::DataFrame> data_buffer(std::max(model_frames, Hip::STREAM_LENGTH));
	ret = load_npy_to_hip_circular_buffer(npy, data_buffer, model_frames);
	if (ret != 0) {
		goto out;
	}

	ret = fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8<Hip::DataFrame>(
		data_buffer,
		static_cast<int8_t*>(input_mems[0]->virt_addr),
		model_frames,
		input_attrs[0].scale,
		input_attrs[0].zp,
		Hip::g_mean_vals,
		Hip::g_std_vals,
		Hip::ModelChannel);
	if (ret < 0) {
		printf("circular buffer preprocess failed! ret=%d\n", ret);
		goto out;
	}
	printf("successfully converted and copied input to memory (circular_buffer path)\n");

	output_mems[0] = rknn_create_mem(ctx, output_attrs[0].size_with_stride);

	ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
	result = new float[4];
	out_fp16 = reinterpret_cast<rknpu2::float16*>(output_mems[0]->virt_addr);
	if (ret < 0) {
		printf("rknn_set_io_mem input fail! ret=%d\n", ret);
		goto out_with_result;
	}

	ret = rknn_set_io_mem(ctx, output_mems[0], &output_attrs[0]);
	if (ret < 0) {
		printf("rknn_set_io_mem output fail! ret=%d\n", ret);
		goto out_with_result;
	}

	ret = rknn_run(ctx, NULL);
	if (ret < 0) {
		printf("rknn run error %d\n", ret);
		goto out_with_result;
	}

	for (int i = 0; i < 4; ++i) {
		result[i] = static_cast<float>(out_fp16[i]);
		printf("out[%d]=%f\n", i, result[i]);
	}
	printf("successfully run\n");
	send_int8_list_adb(reinterpret_cast<int8_t*>(result), 4 * sizeof(float));

out_with_result:
	if (result) {
		delete[] result;
	}
	if (output_mems[0]) {
		rknn_destroy_mem(ctx, output_mems[0]);
	}
out:
	if (input_mems[0]) {
		rknn_destroy_mem(ctx, input_mems[0]);
	}
	rknn_destroy(ctx);
	return ret < 0 ? -1 : 0;
}
