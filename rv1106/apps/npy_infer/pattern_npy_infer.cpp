#include "rknn/rknn_api.h"
#include <float.h>
#include <stdio.h>
#include <vector>
#include "model_process/model_process.h"
#include "npy/npy.h"
#include <array>
#include "fp16/Float16.h"

void send_int8_list_adb(const int8_t* data, size_t size) {
    FILE* tmp = fopen("/tmp/adb_data.bin", "wb");
    if (tmp) {
        fwrite(data, 1, size, tmp);
        fclose(tmp);
        // system("scp /tmp/adb_data.bin sxs@169.254.154.248:/home/sxs/TCN/timesnet_rknn");
        printf("output saved to /tmp/adb_data.bin\n");
    } else {
        printf("failed to write /tmp/adb_data.bin\n");
    }
}

int main(int argc, char** argv) {
    
    if (argc < 3) {
        printf("Usage:%s model_path npy_path\n", argv[0]);
        return -1;
    }
    char *model_path = argv[1];
    char *npy_path = argv[2];
    NpyArray npy;
    if (load_npy(npy_path, npy)) {
        size_t elements = npy.data.size() / npy.elem_size;
        printf("成功读取 %s, dtype=%s, 元素数=%zu, 形状=[", npy_path, npy.dtype.c_str(), elements);
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

    rknn_tensor_mem *input_mems[1];
    input_attrs[0].type = input_type;
    input_attrs[0].fmt = input_layout;
    input_attrs[0].pass_through = 1;
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

    int batch = input_attrs[0].dims[0];
    int channel = npy.shape[1];
    int h = input_attrs[0].dims[2];
    int w = input_attrs[0].dims[3];

    NCHW_float32_to_NC1HWC2_int8(reinterpret_cast<const float*>(npy.data.data()),
                                 static_cast<int8_t*>(input_mems[0]->virt_addr),
                                 batch, channel, h, w, 
                                 input_attrs[0].scale, input_attrs[0].zp,
                                 Hip::g_mean_vals, Hip::g_std_vals, Hip::ModelChannel);
    printf("successfully converted and copied input to memory\n");
    
    rknn_tensor_mem *output_mems[1];

    output_mems[0] = rknn_create_mem(ctx, output_attrs[0].size_with_stride);

    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    float *result = new float[4];
    rknpu2::float16 *out_fp16 = reinterpret_cast<rknpu2::float16*>(output_mems[0]->virt_addr);
    if (ret < 0)
    {
        printf("rknn_set_io_mem input fail! ret=%d\n", ret);
        goto out;
    }

    ret = rknn_set_io_mem(ctx, output_mems[0], &output_attrs[0]);
    if (ret < 0)
    {
        printf("rknn_set_io_mem output fail! ret=%d\n", ret);
        goto out;
    }

    ret = rknn_run(ctx, NULL);
    if (ret < 0)
    {
      printf("rknn run error %d\n", ret);
      goto out;
    }
    
    for (int i = 0; i < 4; ++i) {
        result[i] = (float)out_fp16[i];
        printf("out[%d]=%f\n", i, result[i]);
    }
    printf("successfully run\n");
    send_int8_list_adb((int8_t *)result, 4 * sizeof(float));
out:
    rknn_destroy_mem(ctx, input_mems[0]);
    rknn_destroy_mem(ctx, output_mems[0]);
    rknn_destroy(ctx);
    delete[] result;
    return 0;
}
