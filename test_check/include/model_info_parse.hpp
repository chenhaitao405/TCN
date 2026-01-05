#include "rknn_api.h"
#include <stdio.h>
#include <string.h>
#include <vector>

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

static inline int round_float_to_int(float v)
{
    return (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
}

int NCHW_float32_to_NC1HWC2_int8(const float *src, int8_t *dst, int batch, int channel, int h, int w, int C2, float scale, int zp,
                                 const float *mean, const float *std, int mean_std_len)
{
    if (C2 < 16)
        C2 = 16;
    if (scale <= 0.f)
        return -1;
    int C1 = (channel + C2 - 1) / C2;
    int hw = h * w;

    int8_t zp_clipped = (int8_t)(zp > 127 ? 127 : (zp < -128 ? -128 : zp));
    memset(dst, zp_clipped, batch * C1 * h * w * C2);

    for (int i = 0; i < batch; i++)
    {
        const float *src_batch = src + i * channel * hw;
        int8_t *dst_batch = dst + i * C1 * hw * C2;

        for (int c = 0; c < channel; ++c)
        {
            float mc = (c < mean_std_len) ? mean[c] : 0.f;
            float sc = (c < mean_std_len && std[c] > 0.f) ? std[c] : 1.f;
            float inv = 1.0f / (sc * scale);
            float bias = zp - mc * inv; // 常值折叠: -mean/(std*scale) + zp

            int plane = c / C2;
            int offset = c % C2;
            int8_t *dst_c = dst_batch + plane * hw * C2;
            const float *src_c = src_batch + c * hw;

            for (int cur_h = 0; cur_h < h; ++cur_h)
            {
                for (int cur_w = 0; cur_w < w; ++cur_w)
                {
                    int cur_hw = cur_h * w + cur_w;
                    int q = round_float_to_int(src_c[cur_hw] * inv + bias);
                    if (q > 127)
                        q = 127;
                    if (q < -128)
                        q = -128;
                    dst_c[C2 * cur_hw + offset] = (int8_t)q;
                }
            }
        }
    }
    return 0;
}

int model_info_parse(rknn_context &ctx, char *model_path,
                     std::vector<rknn_tensor_attr> &input_attrs,
                     std::vector<rknn_tensor_attr> &output_attrs)
{
    int ret = rknn_init(&ctx, model_path, 0, 0, NULL);
    if (ret != RKNN_SUCC)
    {
        printf("rknn init failed! ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version sdk_ver;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret != RKNN_SUCC)
    {
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
    input_attrs.assign(io_num.n_input, {});
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_init error! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    printf("output tensors:\n");
    output_attrs.assign(io_num.n_output, {});
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
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

    return RKNN_SUCC;
}