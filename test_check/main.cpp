#include "rknn_api.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdint>

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  char dims[128] = {0};
  for (int i = 0; i < attr->n_dims; ++i)
  {
    int idx = strlen(dims);
    sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
  }
  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride=%d, h_stride=%d, size_with_stride=%d,"
         " pass_through=%hhu, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size, attr->w_stride, attr->h_stride, attr->size_with_stride, 
         attr->pass_through, get_format_string(attr->fmt), get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


struct NpyArray {
    std::vector<size_t> shape;
    std::vector<uint8_t> data;
    size_t elem_size = 0;
    std::string dtype;
};

static bool parse_shape(const std::string& header, std::vector<size_t>& shape) {
    auto shape_pos = header.find("shape");
    if (shape_pos == std::string::npos) return false;
    auto left = header.find('(', shape_pos);
    auto right = header.find(')', left);
    if (left == std::string::npos || right == std::string::npos || right <= left + 1) return false;
    std::string dims = header.substr(left + 1, right - left - 1);
    std::stringstream ss(dims);
    shape.clear();
    while (ss) {
        while (ss.peek() == ' ' || ss.peek() == '\t') ss.get();
        if (ss.peek() == ',') { ss.get(); continue; }
        size_t value;
        if (!(ss >> value)) break;
        shape.push_back(value);
        while (ss.peek() == ' ' || ss.peek() == '\t') ss.get();
        if (ss.peek() == ',') ss.get();
    }
    return !shape.empty();
}

static bool load_npy(const std::string& path, NpyArray& out) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs) {
        fprintf(stderr, "无法打开npy文件: %s\n", path.c_str());
        return false;
    }

    char magic[6];
    fs.read(magic, 6);
    if (fs.gcount() != 6 || std::string(magic, 6) != "\x93NUMPY") {
        fprintf(stderr, "npy文件参数错误: %s\n", path.c_str());
        return false;
    }

    unsigned char major = 0, minor = 0;
    fs.read(reinterpret_cast<char*>(&major), 1);
    fs.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        fs.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else {
        fs.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, ' ');
    fs.read(&header[0], header_len);

    if (header.find("'fortran_order': False") == std::string::npos) {
        fprintf(stderr, "仅支持 C-order npy 文件: %s\n", path.c_str());
        return false;
    }

    auto descr_pos = header.find("'descr':");
    if (descr_pos == std::string::npos) {
        fprintf(stderr, "缺少 descr 字段: %s\n", path.c_str());
        return false;
    }
    auto first_quote = header.find('\'', descr_pos + 8);
    auto second_quote = header.find('\'', first_quote + 1);
    if (first_quote == std::string::npos || second_quote == std::string::npos) {
        fprintf(stderr, "解析 descr 失败: %s\n", path.c_str());
        return false;
    }
    std::string descr = header.substr(first_quote + 1, second_quote - first_quote - 1);

    bool is_float32 = descr.find("f4") != std::string::npos;
    bool is_uint8 = descr.find("u1") != std::string::npos;
    if (!is_float32 && !is_uint8) {
        fprintf(stderr, "仅支持 float32 或 uint8 的npy: %s\n", path.c_str());
        return false;
    }

    size_t elem_size = is_float32 ? sizeof(float) : sizeof(uint8_t);
    bool need_swap = (descr[0] == '>' && elem_size > 1);

    std::vector<size_t> shape;
    if (!parse_shape(header, shape)) {
        fprintf(stderr, "解析 shape 失败: %s\n", path.c_str());
        return false;
    }

    size_t total = 1;
    for (size_t dim : shape) total *= dim;
    if (total == 0) {
        fprintf(stderr, "npy 数据元素数量为0: %s\n", path.c_str());
        return false;
    }

    std::vector<uint8_t> raw(total * elem_size);
    fs.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (static_cast<size_t>(fs.gcount()) != raw.size()) {
        fprintf(stderr, "读取npy数据失败: %s\n", path.c_str());
        return false;
    }

    if (need_swap && elem_size == sizeof(float)) {
        for (size_t i = 0; i < total; ++i) {
            uint8_t* elem = raw.data() + i * elem_size;
            uint8_t t0 = elem[0]; elem[0] = elem[3]; elem[3] = t0;
            uint8_t t1 = elem[1]; elem[1] = elem[2]; elem[2] = t1;
        }
    }

    out.shape = std::move(shape);
    out.data = std::move(raw);
    out.elem_size = elem_size;
    out.dtype = is_float32 ? "float32" : "uint8";
    return true;
}

/*
 * 将NCHW uint8格式转换为NC1HWC2 int8格式
 * src: NCHW格式的uint8数据
 * dst: NC1HWC2格式的int8数据
 * batch, channel, h, w: NCHW的shape信息
 * C2: NC1HWC2的C2维度，通常为16
 */
int NCHW_uint8_to_NC1HWC2_int8(const uint8_t* src, int8_t* dst, int batch, int channel, int h, int w, int C2)
{
    int C1 = (channel + C2 - 1) / C2;  // 向上取整
    int hw = h * w;
    
    // 初始化dst为0
    memset(dst, 0, batch * C1 * h * w * C2);
    
    for (int i = 0; i < batch; i++) {
        const uint8_t* src_batch = src + i * channel * hw;
        int8_t* dst_batch = dst + i * C1 * hw * C2;
        
        for (int c = 0; c < channel; ++c) {
            int plane = c / C2;
            int offset = c % C2;
            int8_t* dst_c = dst_batch + plane * hw * C2;
            const uint8_t* src_c = src_batch + c * hw;
            
            for (int cur_h = 0; cur_h < h; ++cur_h) {
                for (int cur_w = 0; cur_w < w; ++cur_w) {
                    int cur_hw = cur_h * w + cur_w;
                    // uint8转int8: 减128
                    dst_c[C2 * cur_hw + offset] = (int8_t)(src_c[cur_hw] - 128);
                }
            }
        }
    }
    return 0;
}

/*
 * 将NCHW float32量化并转换为NC1HWC2 int8格式
 * 量化公式: q = round(x / scale) + zp, 再裁剪到[-128, 127]
 * src: NCHW格式的float32数据
 * dst: NC1HWC2格式的int8数据
 * batch, channel, h, w: NCHW的shape信息
 * C2: NC1HWC2的C2维度，最小为16
 * scale: 量化scale（>0）
 * zp: 量化zero-point（int）
 */

// 兼容ARMv7的浮点四舍五入到int的辅助函数（不依赖lroundf）
static inline int round_float_to_int(float v) {
    return (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
}

static const float g_mean_vals[8] = {0.912629f, -0.820086f, 0.197228f, 3.041710f, 8.390683f, -0.177071f, -33.347044f, 2.272569f};
static const float g_std_vals[8]  = {16.687198f, 29.007817f, 82.961570f, 4.110988f, 4.230163f, 1.639207f, 30.426489f, 113.209047f};

int NCHW_float32_to_NC1HWC2_int8(const float* src, int8_t* dst, int batch, int channel, int h, int w, int C2, float scale, int zp,
                                 const float* mean, const float* std, int mean_std_len)
{
    if (C2 < 16) C2 = 16;
    if (scale <= 0.f) return -1;
    int C1 = (channel + C2 - 1) / C2;
    int hw = h * w;

    int8_t zp_clipped = (int8_t)(zp > 127 ? 127 : (zp < -128 ? -128 : zp));
    memset(dst, zp_clipped, batch * C1 * h * w * C2);

    for (int i = 0; i < batch; i++) {
        const float* src_batch = src + i * channel * hw;
        int8_t* dst_batch = dst + i * C1 * hw * C2;

        for (int c = 0; c < channel; ++c) {
            float mc = (c < mean_std_len) ? mean[c] : 0.f;
            float sc = (c < mean_std_len && std[c] > 0.f) ? std[c] : 1.f;
            float inv = 1.0f / (sc * scale);
            float bias = zp - mc * inv;  // 常值折叠: -mean/(std*scale) + zp

            int plane = c / C2;
            int offset = c % C2;
            int8_t* dst_c = dst_batch + plane * hw * C2;
            const float* src_c = src_batch + c * hw;

            for (int cur_h = 0; cur_h < h; ++cur_h) {
                for (int cur_w = 0; cur_w < w; ++cur_w) {
                    int cur_hw = cur_h * w + cur_w;
                    int q = round_float_to_int(src_c[cur_hw] * inv + bias);
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    dst_c[C2 * cur_hw + offset] = (int8_t)q;
                }
            }
        }
    }
    return 0;
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
    int ret = rknn_init(&ctx, model_path, 0, 0, NULL);

    if (ret < 0) {
        printf("rknn init failed! ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version sdk_ver;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
        printf("rknn_init error! ret=%d\n", ret);
        return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_output);
    for (uint32_t i=0; i< io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    rknn_custom_string custom_string;
    ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("custom string: %s\n", custom_string.string);

    rknn_tensor_type input_type = RKNN_TENSOR_INT8;
    rknn_tensor_format input_layout = RKNN_TENSOR_NC1HWC2;

    rknn_tensor_mem *input_mems[1];
    input_attrs[0].type = input_type;
    input_attrs[0].fmt = input_layout;
    input_attrs[0].pass_through = 1;
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

    // 转换数据格式: NCHW float32 -> NC1HWC2 int8（含归一化与量化折叠）
    int batch = input_attrs->dims[0];
    int channel = npy.shape[1];
    int h = input_attrs->dims[2];
    int w = input_attrs->dims[3];
    int C2 = input_attrs->dims[4];

    NCHW_float32_to_NC1HWC2_int8(reinterpret_cast<const float*>(npy.data.data()),
                                 static_cast<int8_t*>(input_mems[0]->virt_addr),
                                 batch, channel, h, w, C2,
                                 0.133385f, -3,
                                 g_mean_vals, g_std_vals, 8);
    printf("successfully converted and copied input to memory\n");
    
    rknn_tensor_mem *output_mems[1];
    output_attrs[0].pass_through = 0;

    output_mems[0] = rknn_create_mem(ctx, output_attrs[0].size_with_stride);

    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
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
    printf("successfully run\n");
    printf("last element is %d\n", ((int8_t *)output_mems[0]->virt_addr)[279*16]);
out:
    rknn_destroy_mem(ctx, input_mems[0]);
    rknn_destroy_mem(ctx, output_mems[0]);
    rknn_destroy(ctx);
    return 0;
}
