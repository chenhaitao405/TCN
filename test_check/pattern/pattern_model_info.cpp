#include "rknn_api.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "model_info_parse.hpp"

static void resource_destroy(rknn_context ctx, rknn_tensor_mem* input_mem, rknn_tensor_mem* output_mem) {
    if (input_mem) rknn_destroy_mem(ctx, input_mem);
    if (output_mem) rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
}

// Convert IEEE 754 half precision to single precision for logging on hosts without native fp16.
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f_exp;
    uint32_t f_mant;
    if (exp == 0) {
        if (mant == 0) {
            f_exp = 0;
            f_mant = 0;
        } else {
            // Subnormal half, normalize it.
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FF;
            f_exp = exp + (127 - 15);
            f_mant = mant << 13;
        }
    } else if (exp == 0x1F) {
        // Inf/NaN preserve payload where possible.
        f_exp = 255;
        f_mant = mant << 13;
    } else {
        f_exp = exp + (127 - 15);
        f_mant = mant << 13;
    }

    uint32_t f_bits = (sign << 31) | (f_exp << 23) | f_mant;
    float out;
    memcpy(&out, &f_bits, sizeof(out));
    return out;
}

static const float g_mean_vals[8] = {
    0.0f, 0.0f, 0.0f,   // gyro
    0.0f, 0.0f, 0.0f,   // acc
    0.0f, 0.0f          // motor
};
static const float g_std_vals[8] = {
    1.0f, 1.0f, 1.0f, // gyro
    1.0f, 1.0f, 1.0f, // acc
    1.0f, 1.0f        // motor
};

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("need model_path\n");
        return 1;
    }
    const char *model_path = argv[1];
    rknn_context ctx;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int ret = model_info_parse(ctx, const_cast<char*>(model_path), input_attrs, output_attrs);

    if (ret != RKNN_SUCC) {
        printf("model info parse failed!\n");
        return -1;
    }

    input_attrs[0].pass_through = 1;
    int h = input_attrs[0].dims[2];
    int w = input_attrs[0].dims[3];
    int expected_frames = h * w;

    rknn_tensor_mem *input_mems[1];
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    rknn_tensor_mem *output_mems[1];
    output_mems[0] = rknn_create_mem(ctx, output_attrs[0].size_with_stride);

    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    if (ret != RKNN_SUCC) {
        printf("rknn_set_io_mem input fail! ret=%d\n", ret);
        resource_destroy(ctx, input_mems[0], output_mems[0]);
        return -1;
    }

    ret = rknn_set_io_mem(ctx, output_mems[0], &output_attrs[0]);
    if (ret != RKNN_SUCC) {
        printf("rknn_set_io_mem output fail! ret=%d\n", ret);
        resource_destroy(ctx, input_mems[0], output_mems[0]);
        return -1;
    }

    float *inputs = new float[expected_frames * 8];
    NCHW_float32_to_NC1HWC2_int8(inputs, (int8_t*)input_mems[0]->virt_addr, 1, 8, 
                            h, w, input_attrs[0].scale, input_attrs[0].zp, g_mean_vals, g_std_vals, 8);
    ret = rknn_run(ctx, NULL);
    if (ret != RKNN_SUCC) {
        printf("rknn run error %d\n", ret);
        resource_destroy(ctx, input_mems[0], output_mems[0]);
        return -1;
    }
    uint16_t *out_fp16 = reinterpret_cast<uint16_t*>(output_mems[0]->virt_addr);
    for (int i = 0; i < 4; ++i) {
        float val = fp16_to_fp32(out_fp16[i]);
        printf("out[%d]=%f\n", i, val);
    }
    resource_destroy(ctx, input_mems[0], output_mems[0]);
    delete[] inputs;
    return 0;
}