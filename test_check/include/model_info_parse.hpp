#include "rknn_api.h"
#include "data_stream.h"
#include <stdio.h>
#include <string.h>
#include <vector>
#include <arm_neon.h>
#define ALIGNED_CHANNEL 16

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

int NCHW_float32_to_NC1HWC2_int8(const float *src, int8_t *dst, int batch, int channel, int h, int w, float scale, int zp,
                                 const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f)
        return -1;
    int C1 = (channel + ALIGNED_CHANNEL - 1) / ALIGNED_CHANNEL;
    int hw = h * w;

    int8_t zp_clipped = (int8_t)(zp > 127 ? 127 : (zp < -128 ? -128 : zp));
    memset(dst, zp_clipped, batch * C1 * h * w * ALIGNED_CHANNEL);

    // NEON：针对channel == 8且ALIGNED_CHANNEL == 16
    if (channel == 8 && ALIGNED_CHANNEL == 16) {
        float inv_arr[8];
        float bias_arr[8];
        for (int c = 0; c < 8; ++c) {
            float mc = (c < mean_std_len) ? mean[c] : 0.f;
            float sc = (c < mean_std_len && std[c] > 0.f) ? std[c] : 1.f;
            inv_arr[c] = 1.0f / (sc * scale);
            bias_arr[c] = zp - mc * inv_arr[c];
        }

        const float32x4_t inv_l = vld1q_f32(inv_arr);
        const float32x4_t inv_h = vld1q_f32(inv_arr + 4);
        const float32x4_t bias_l = vld1q_f32(bias_arr);
        const float32x4_t bias_h = vld1q_f32(bias_arr + 4);
        const float32x4_t v_zero = vdupq_n_f32(0.0f);
        const float32x4_t v_0_5 = vdupq_n_f32(0.5f);

        for (int i = 0; i < batch; ++i) {
            const float *src_batch = src + i * channel * hw;
            int8_t *dst_batch = dst + i * C1 * hw * ALIGNED_CHANNEL; // C1==1

            for (int cur_h = 0; cur_h < h; ++cur_h) {
                for (int cur_w = 0; cur_w < w; ++cur_w) {
                    int idx = cur_h * w + cur_w;

                    // 组装当前像素的8通道值到两个向量
                    float32x4_t v_l;
                    float32x4_t v_h;
                    v_l = vsetq_lane_f32(*(src_batch + 0 * hw + idx), vdupq_n_f32(0.f), 0);
                    v_l = vsetq_lane_f32(*(src_batch + 1 * hw + idx), v_l, 1);
                    v_l = vsetq_lane_f32(*(src_batch + 2 * hw + idx), v_l, 2);
                    v_l = vsetq_lane_f32(*(src_batch + 3 * hw + idx), v_l, 3);
                    v_h = vsetq_lane_f32(*(src_batch + 4 * hw + idx), vdupq_n_f32(0.f), 0);
                    v_h = vsetq_lane_f32(*(src_batch + 5 * hw + idx), v_h, 1);
                    v_h = vsetq_lane_f32(*(src_batch + 6 * hw + idx), v_h, 2);
                    v_h = vsetq_lane_f32(*(src_batch + 7 * hw + idx), v_h, 3);

                    // 归一化与偏置：x*inv + bias
                    v_l = vfmaq_f32(bias_l, inv_l, v_l);
                    v_h = vfmaq_f32(bias_h, inv_h, v_h);

                    // 四舍五入到最近整数
                    uint32x4_t m_l = vcltq_f32(v_l, v_zero);
                    uint32x4_t m_h = vcltq_f32(v_h, v_zero);
                    v_l = vaddq_f32(v_l, vbslq_f32(m_l, vnegq_f32(v_0_5), v_0_5));
                    v_h = vaddq_f32(v_h, vbslq_f32(m_h, vnegq_f32(v_0_5), v_0_5));

                    // 转int8并饱和
                    int16x8_t s16_vec8 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_l)), vqmovn_s32(vcvtq_s32_f32(v_h)));
                    int8x8_t res8 = vqmovn_s16(s16_vec8);

                    // 写入当前像素的 contiguous C2(=16) block的前8通道
                    int8_t *p_dst_pixel = dst_batch + ALIGNED_CHANNEL * idx;
                    vst1_s8(p_dst_pixel, res8);
                }
            }
        }
        return 0;
    }

    // 通用回退路径：逐通道逐元素标量处理
    for (int i = 0; i < batch; ++i)
    {
        const float *src_batch = src + i * channel * hw;
        int8_t *dst_batch = dst + i * C1 * hw * ALIGNED_CHANNEL;

        for (int c = 0; c < channel; ++c)
        {
            float mc = (c < mean_std_len) ? mean[c] : 0.f;
            float sc = (c < mean_std_len && std[c] > 0.f) ? std[c] : 1.f;
            float inv = 1.0f / (sc * scale);
            float bias = zp - mc * inv;

            int plane = c / ALIGNED_CHANNEL;
            int offset = c % ALIGNED_CHANNEL;
            int8_t *dst_c = dst_batch + plane * hw * ALIGNED_CHANNEL;
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
                    dst_c[ALIGNED_CHANNEL * cur_hw + offset] = (int8_t)q;
                }
            }
        }
    }
    return 0;
}


// asume N=H=1, C=8, C2=16, C1=1
static inline int fast_NHWC_float32_deque_to_NC1HWC2_int8(const DataStream &src, int8_t *dst,  int W, 
                                    float scale, int zp, const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f) return -1;
    int offset = src.size() - W;
    if (offset < 0) return -1;

    // const param
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_0_5 = vdupq_n_f32(0.5f);
    int8_t zp_clipped = (int8_t)(zp > 127 ? 127 : (zp < -128 ? -128 : zp));
    memset(dst, zp_clipped, W * ALIGNED_CHANNEL);
    
    float inv[8];
    float bias[8];
    for (int c = 0; c < 8; ++c) {
        inv[c] = 1.0f / (std[c] * scale);
        bias[c] = (float)zp - mean[c] * inv[c];
    }

    const float32x4_t inv_l = vld1q_f32(inv);
    const float32x4_t inv_h = vld1q_f32(inv + 4);
    const float32x4_t bias_l = vld1q_f32(bias);
    const float32x4_t bias_h = vld1q_f32(bias + 4);
    int w = 0;
    for (; w <= W - 2; w += 2) {
        // load data
        const DataFrame &x0 = src[offset + w];
        const DataFrame &x1 = src[offset + w + 1];

        float32x4_t v0_l = vld1q_f32((const float*)&x0);
        float32x4_t v0_h = vld1q_f32((const float*)&x0 + 4);

        float32x4_t v1_l = vld1q_f32((const float*)&x1);
        float32x4_t v1_h = vld1q_f32((const float*)&x1 + 4);

        // 归一化计算
        v0_l = vfmaq_f32(bias_l, inv_l, v0_l);
        v0_h = vfmaq_f32(bias_h, inv_h, v0_h);
        v1_l = vfmaq_f32(bias_l, inv_l, v1_l);
        v1_h = vfmaq_f32(bias_h, inv_h, v1_h);

        // Rounding
        uint32x4_t m0_l = vcltq_f32(v0_l, v_zero);
        uint32x4_t m0_h = vcltq_f32(v0_h, v_zero);
        uint32x4_t m1_l = vcltq_f32(v1_l, v_zero);
        uint32x4_t m1_h = vcltq_f32(v1_h, v_zero);

        v0_l = vaddq_f32(v0_l, vbslq_f32(m0_l, vnegq_f32(v_0_5), v_0_5));
        v0_h = vaddq_f32(v0_h, vbslq_f32(m0_h, vnegq_f32(v_0_5), v_0_5));
        v1_l = vaddq_f32(v1_l, vbslq_f32(m1_l, vnegq_f32(v_0_5), v_0_5));
        v1_h = vaddq_f32(v1_h, vbslq_f32(m1_h, vnegq_f32(v_0_5), v_0_5));

        // Saturating Narrowing
        int16x8_t s16_vec8_0 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v0_l)), vqmovn_s32(vcvtq_s32_f32(v0_h)));
        int8x8_t res8_0 = vqmovn_s16(s16_vec8_0);

        int16x8_t s16_vec8_1 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v1_l)), vqmovn_s32(vcvtq_s32_f32(v1_h)));
        int8x8_t res8_1 = vqmovn_s16(s16_vec8_1);

        // Store
        int8_t *p_dst = dst + w * 16;

        vst1_s8(p_dst, res8_0);      
        vst1_s8(p_dst + 16, res8_1);
    }

    for (; w < W; ++w) {
        const DataFrame &x = src[offset + w];
        float32x4_t v_l = vld1q_f32((const float*)&x);
        float32x4_t v_h = vld1q_f32((const float*)&x + 4);
        v_l = vfmaq_f32(bias_l, inv_l, v_l);
        v_h = vfmaq_f32(bias_h, inv_h, v_h);
        
        uint32x4_t m_l = vcltq_f32(v_l, v_zero);
        uint32x4_t m_h = vcltq_f32(v_h, v_zero);
        v_l = vaddq_f32(v_l, vbslq_f32(m_l, vnegq_f32(v_0_5), v_0_5));
        v_h = vaddq_f32(v_h, vbslq_f32(m_h, vnegq_f32(v_0_5), v_0_5));

        int8x8_t res8 = vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_l)), vqmovn_s32(vcvtq_s32_f32(v_h))));
        vst1_s8(dst + w * 16, res8);
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
            printf("rknn_query error! ret=%d\n", ret);
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