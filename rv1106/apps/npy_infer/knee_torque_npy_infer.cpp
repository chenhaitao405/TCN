#include "rknn/rknn_api.h"
#include <stdio.h>
#include <vector>
#include "model_process/model_process.h"
#include "npy/npy.h"
#include "fp16/Float16.h"


static int preprocess_input_by_type(const float* src, void* dst, const rknn_tensor_attr& input_attr,
                                    int batch, int channel, int h, int w)
{
    if (input_attr.type == RKNN_TENSOR_INT8)
    {
        return NCHW_float32_to_NC1HWC2_int8(
            src,
            static_cast<int8_t*>(dst),
            batch, channel, h, w,
            input_attr.scale, input_attr.zp,
            Knee::g_mean_vals, Knee::g_std_vals, Knee::ModelChannel);
    }

    if (input_attr.type == RKNN_TENSOR_INT16)
    {
        return NCHW_float32_to_NC1HWC2_int16(
            src,
            static_cast<int16_t*>(dst),
            batch, channel, h, w,
            input_attr.scale, input_attr.zp,
            Knee::g_mean_vals, Knee::g_std_vals, Knee::ModelChannel);
    }

    printf("Unsupported input tensor type: %s\n", get_type_string(input_attr.type));
    return -1;
}

static bool dump_output_as_float32_series(const rknn_tensor_attr& output_attr, void* output_addr, const char* file_path)
{
    const int W = (output_attr.n_dims >= 4) ? output_attr.dims[3] : output_attr.n_elems;
    const int C2 = (output_attr.n_dims >= 5) ? output_attr.dims[4] : 1;
    if (W <= 0 || C2 <= 0)
    {
        printf("invalid output dims: n_dims=%d, W=%d, C2=%d\n", output_attr.n_dims, W, C2);
        return false;
    }

    std::vector<float> torque_series(W, 0.0f);
    if (output_attr.type == RKNN_TENSOR_INT8)
    {
        int8_t* p = static_cast<int8_t*>(output_addr);
        for (int i = 0; i < W; ++i)
            torque_series[i] = (static_cast<int>(p[i * C2]) - output_attr.zp) * output_attr.scale;
    }
    else if (output_attr.type == RKNN_TENSOR_INT16)
    {
        int16_t* p = static_cast<int16_t*>(output_addr);
        for (int i = 0; i < W; ++i)
            torque_series[i] = (static_cast<int>(p[i * C2]) - output_attr.zp) * output_attr.scale;
    }
    else if (output_attr.type == RKNN_TENSOR_FLOAT16)
    {
        rknpu2::float16* p = static_cast<rknpu2::float16*>(output_addr);
        for (int i = 0; i < W; ++i)
            torque_series[i] = static_cast<float>(p[i * C2]);
    }
    else if (output_attr.type == RKNN_TENSOR_FLOAT32)
    {
        float* p = static_cast<float*>(output_addr);
        for (int i = 0; i < W; ++i)
            torque_series[i] = p[i * C2];
    }
    else
    {
        printf("Unsupported output tensor type: %s\n", get_type_string(output_attr.type));
        return false;
    }

    printf("output[last]=%.6f\n", torque_series[W - 1]);

    FILE* tmp = fopen(file_path, "wb");
    if (!tmp)
    {
        printf("failed to write %s\n", file_path);
        return false;
    }
    fwrite(torque_series.data(), sizeof(float), torque_series.size(), tmp);
    fclose(tmp);
    printf("output saved to %s, len=%d\n", file_path, W);
    return true;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage:%s model_path npy_path\n", argv[0]);
        return -1;
    }

    char* model_path = argv[1];
    char* npy_path = argv[2];

    NpyArray npy;
    if (!load_npy(npy_path, npy)) {
        printf("failed to read npy file: %s\n", npy_path);
        return -1;
    }

    size_t elements = npy.data.size() / npy.elem_size;
    printf("成功读取 %s, dtype=%s, 元素数=%zu, 形状=[", npy_path, npy.dtype.c_str(), elements);
    for (size_t i = 0; i < npy.shape.size(); ++i) {
        printf("%zu%s", npy.shape[i], (i + 1 == npy.shape.size()) ? "" : ", ");
    }
    printf("]\n");

    if ((npy.dtype != "float32" && npy.dtype != "f4") || npy.shape.size() < 4) {
        printf("expect npy dtype=float32 and shape like [N,C,H,W], got dtype=%s, rank=%zu\n",
               npy.dtype.c_str(), npy.shape.size());
        return -1;
    }

    rknn_context ctx;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int ret = model_info_parse(ctx, model_path, input_attrs, output_attrs);
    if (ret != RKNN_SUCC) {
        printf("model info parse failed! ret=%d\n", ret);
        return -1;
    }

    if (input_attrs.empty() || output_attrs.empty()) {
        printf("invalid model io count\n");
        rknn_destroy(ctx);
        return -1;
    }

    if (!Knee::checkModelInfo(input_attrs)) {
        rknn_destroy(ctx);
        return -1;
    }
    input_attrs[0].pass_through = 1;
    // output_attrs[0].pass_through = 1;
    // if (input_attrs[0].type == RKNN_TENSOR_INT16) {
    //     input_attrs[0].dims[1] = 4;
    //     input_attrs[0].n_elems = 2240*4;
    //     input_attrs[0].size = 4480*4;
    //     input_attrs[0].size_with_stride = 4480*4;
    //     output_attrs[0].dims[1] = 8;
    //     output_attrs[0].n_elems = 4480 * 4;
    //     output_attrs[0].size = 8960 * 4;
    //     output_attrs[0].size_with_stride = 8960 * 4;
    // }
    
    std::vector<rknn_tensor_mem*> input_mems;
    std::vector<rknn_tensor_mem*> output_mems;
    ret = alllocate_set_io_memory(ctx, input_attrs, output_attrs, input_mems, output_mems);
    if (ret != RKNN_SUCC) {
        printf("alllocate_set_io_memory fail! ret=%d\n", ret);
        resource_destroy(ctx, input_mems, output_mems);
        return -1;
    }

    int batch = input_attrs[0].dims[0];
    int channel = (int)npy.shape[1];
    int h = input_attrs[0].dims[2];
    int w = input_attrs[0].dims[3];

    ret = preprocess_input_by_type(
        reinterpret_cast<const float*>(npy.data.data()),
        input_mems[0]->virt_addr,
        input_attrs[0],
        batch, channel, h, w);

    if (ret != 0) {
        printf("input preprocess failed! ret=%d\n", ret);
        resource_destroy(ctx, input_mems, output_mems);
        return -1;
    }

    ret = rknn_run(ctx, NULL);
    if (ret != RKNN_SUCC) {
        printf("rknn run error %d\n", ret);
        resource_destroy(ctx, input_mems, output_mems);
        return -1;
    }

    printf("successfully run\n");
    if (!dump_output_as_float32_series(output_attrs[0], output_mems[0]->virt_addr, "/tmp/adb_data.bin")) {
        resource_destroy(ctx, input_mems, output_mems);
        return -1;
    }

    resource_destroy(ctx, input_mems, output_mems);
    return 0;
}
