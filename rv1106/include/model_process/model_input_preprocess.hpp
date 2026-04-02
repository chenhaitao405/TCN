#pragma once
#include "data_stream.h"
#include <deque>
#include <string.h>
#include <arm_neon.h>
#include "boost/circular_buffer.hpp"

constexpr size_t ALIGNED_BYTES = 16;
constexpr size_t ALIGNED_CHANNEL_INT8  = ALIGNED_BYTES / sizeof(int8_t);
constexpr size_t ALIGNED_CHANNEL_INT16 = ALIGNED_BYTES / sizeof(int16_t);

static inline int round_float_to_int(float v)
{
    return (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
}

int NCHW_float32_to_NC1HWC2_int8(const float *src, int8_t *dst, int batch, int channel, int h, int w, float scale, int zp,
                                 const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f)
        return -1;
    int C1 = (channel + ALIGNED_CHANNEL_INT8 - 1) / ALIGNED_CHANNEL_INT8;
    int hw = h * w;

    int8_t zp_clipped = (int8_t)(zp > 127 ? 127 : (zp < -128 ? -128 : zp));
    memset(dst, zp_clipped, batch * C1 * h * w * ALIGNED_CHANNEL_INT8);

    // NEON：针对channel == 8且ALIGNED_CHANNEL == 16
    if (channel == 8 && ALIGNED_CHANNEL_INT8 == 16) {
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
            int8_t *dst_batch = dst + i * C1 * hw * ALIGNED_CHANNEL_INT8; // C1==1

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
                    int8_t *p_dst_pixel = dst_batch + ALIGNED_CHANNEL_INT8 * idx;
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
        int8_t *dst_batch = dst + i * C1 * hw * ALIGNED_CHANNEL_INT8;

        for (int c = 0; c < channel; ++c)
        {
            float mc = (c < mean_std_len) ? mean[c] : 0.f;
            float sc = (c < mean_std_len && std[c] > 0.f) ? std[c] : 1.f;
            float inv = 1.0f / (sc * scale);
            float bias = zp - mc * inv;

            int plane = c / ALIGNED_CHANNEL_INT8;
            int offset = c % ALIGNED_CHANNEL_INT8;
            int8_t *dst_c = dst_batch + plane * hw * ALIGNED_CHANNEL_INT8;
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
                    dst_c[ALIGNED_CHANNEL_INT8 * cur_hw + offset] = (int8_t)q;
                }
            }
        }
    }
    return 0;
}

int NCHW_float32_to_NC1HWC2_int16(const float *src, int16_t *dst, int batch, int channel, int h, int w, float scale, int zp,
                                  const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f)
        return -1;

    const int C2 = ALIGNED_CHANNEL_INT16;
    int C1 = (channel + C2 - 1) / C2;
    int hw = h * w;

    int16_t zp_clipped = (int16_t)(zp > 32767 ? 32767 : (zp < -32768 ? -32768 : zp));
    int total = batch * C1 * h * w * C2;
    for (int i = 0; i < total; ++i)
        dst[i] = zp_clipped;

    for (int i = 0; i < batch; ++i)
    {
        const float *src_batch = src + i * channel * hw;
        int16_t *dst_batch = dst + i * C1 * hw * C2;

        for (int c = 0; c < channel; ++c)
        {
            float mc = (c < mean_std_len) ? mean[c] : 0.f;
            float sc = (c < mean_std_len && std[c] > 0.f) ? std[c] : 1.f;
            float inv = 1.0f / (sc * scale);
            float bias = zp - mc * inv;

            int plane = c / C2;
            int offset = c % C2;
            int16_t *dst_c = dst_batch + plane * hw * C2;
            const float *src_c = src_batch + c * hw;

            for (int cur_h = 0; cur_h < h; ++cur_h)
            {
                for (int cur_w = 0; cur_w < w; ++cur_w)
                {
                    int cur_hw = cur_h * w + cur_w;
                    int q = round_float_to_int(src_c[cur_hw] * inv + bias);
                    if (q > 32767)
                        q = 32767;
                    if (q < -32768)
                        q = -32768;
                    dst_c[C2 * cur_hw + offset] = (int16_t)q;
                }
            }
        }
    }

    return 0;
}


// asume N=H=1, C=8, C2=16, C1=1
template<typename T>
static inline int fast_NHWC_float32_deque_to_NC1HWC2_int8(const std::deque<T> &src, int8_t *dst,  int W, 
                                    float scale, int zp, const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f) return -1;
    int offset = src.size() - W;
    if (offset < 0) return -1;

    // const param
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_0_5 = vdupq_n_f32(0.5f);
    int8_t zp_clipped = (int8_t)(zp > 127 ? 127 : (zp < -128 ? -128 : zp));
    memset(dst, zp_clipped, W * ALIGNED_CHANNEL_INT8);
    
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
        const T &x0 = src[offset + w];
        const T &x1 = src[offset + w + 1];

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
        const T &x = src[offset + w];
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

// asume N=H=1, C=8, C2=8, C1=1
template<typename T>
static inline int fast_NHWC_float32_deque_to_NC1HWC2_int16(const std::deque<T> &src, int16_t *dst,  int W, 
                                    float scale, int zp, const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f) return -1;
    int offset = src.size() - W;
    if (offset < 0) return -1;

    // const param
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_0_5 = vdupq_n_f32(0.5f);
    
    float inv[8];
    float bias[8];
    for (int c = 0; c < 8; ++c) {
        float mc = (c < mean_std_len) ? mean[c] : 0.f;
        float sc = (c < mean_std_len && std[c] > 0.f) ? std[c] : 1.f;
        inv[c] = 1.0f / (sc * scale);
        bias[c] = (float)zp - mc * inv[c];
    }

    const float32x4_t inv_l = vld1q_f32(inv);
    const float32x4_t inv_h = vld1q_f32(inv + 4);
    const float32x4_t bias_l = vld1q_f32(bias);
    const float32x4_t bias_h = vld1q_f32(bias + 4);
    int w = 0;
    for (; w <= W - 2; w += 2) {
        // load data
        const T &x0 = src[offset + w];
        const T &x1 = src[offset + w + 1];

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

        // Saturating Narrowing to int16
        int16x8_t res16_0 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v0_l)), vqmovn_s32(vcvtq_s32_f32(v0_h)));
        int16x8_t res16_1 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v1_l)), vqmovn_s32(vcvtq_s32_f32(v1_h)));

        // Store
        int16_t *p_dst = dst + w * 8;

        vst1q_s16(p_dst, res16_0);
        vst1q_s16(p_dst + 8, res16_1);
    }

    for (; w < W; ++w) {
        const T &x = src[offset + w];
        float32x4_t v_l = vld1q_f32((const float*)&x);
        float32x4_t v_h = vld1q_f32((const float*)&x + 4);
        v_l = vfmaq_f32(bias_l, inv_l, v_l);
        v_h = vfmaq_f32(bias_h, inv_h, v_h);
        
        uint32x4_t m_l = vcltq_f32(v_l, v_zero);
        uint32x4_t m_h = vcltq_f32(v_h, v_zero);
        v_l = vaddq_f32(v_l, vbslq_f32(m_l, vnegq_f32(v_0_5), v_0_5));
        v_h = vaddq_f32(v_h, vbslq_f32(m_h, vnegq_f32(v_0_5), v_0_5));

        int16x8_t res16 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_l)), vqmovn_s32(vcvtq_s32_f32(v_h)));
        vst1q_s16(dst + w * 8, res16);
    }

    return 0;
}

// asume N=H=1, C=10, C2=16, C1=1
template<typename T>
static inline int fast_NHWC_float32_deque_to_NC1HWC2_int8_C10(const std::deque<T> &src, int8_t *dst, int W,
                                    float scale, int zp, const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f) return -1;
    int offset = src.size() - W;
    if (offset < 0) return -1;

    // const param
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_0_5 = vdupq_n_f32(0.5f);
    int8_t zp_clipped = (int8_t)(zp > 127 ? 127 : (zp < -128 ? -128 : zp));
    memset(dst, zp_clipped, W * ALIGNED_CHANNEL_INT8);

    float inv[10];
    float bias[10];
    for (int c = 0; c < 10; ++c) {
        float mc = (c < mean_std_len) ? mean[c] : 0.f;
        float sc = (c < mean_std_len && std[c] > 0.f) ? std[c] : 1.f;
        inv[c] = 1.0f / (sc * scale);
        bias[c] = (float)zp - mc * inv[c];
    }

    const float32x4_t inv_l = vld1q_f32(inv);
    const float32x4_t inv_h = vld1q_f32(inv + 4);
    const float32x4_t bias_l = vld1q_f32(bias);
    const float32x4_t bias_h = vld1q_f32(bias + 4);

    int w = 0;
    for (; w <= W - 2; w += 2) {
        const T &x0 = src[offset + w];
        const T &x1 = src[offset + w + 1];

        float32x4_t v0_l = vld1q_f32((const float*)&x0);
        float32x4_t v0_h = vld1q_f32((const float*)&x0 + 4);
        float32x4_t v1_l = vld1q_f32((const float*)&x1);
        float32x4_t v1_h = vld1q_f32((const float*)&x1 + 4);

        v0_l = vfmaq_f32(bias_l, inv_l, v0_l);
        v0_h = vfmaq_f32(bias_h, inv_h, v0_h);
        v1_l = vfmaq_f32(bias_l, inv_l, v1_l);
        v1_h = vfmaq_f32(bias_h, inv_h, v1_h);

        uint32x4_t m0_l = vcltq_f32(v0_l, v_zero);
        uint32x4_t m0_h = vcltq_f32(v0_h, v_zero);
        uint32x4_t m1_l = vcltq_f32(v1_l, v_zero);
        uint32x4_t m1_h = vcltq_f32(v1_h, v_zero);

        v0_l = vaddq_f32(v0_l, vbslq_f32(m0_l, vnegq_f32(v_0_5), v_0_5));
        v0_h = vaddq_f32(v0_h, vbslq_f32(m0_h, vnegq_f32(v_0_5), v_0_5));
        v1_l = vaddq_f32(v1_l, vbslq_f32(m1_l, vnegq_f32(v_0_5), v_0_5));
        v1_h = vaddq_f32(v1_h, vbslq_f32(m1_h, vnegq_f32(v_0_5), v_0_5));

        int8x8_t res8_0 = vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v0_l)), vqmovn_s32(vcvtq_s32_f32(v0_h))));
        int8x8_t res8_1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v1_l)), vqmovn_s32(vcvtq_s32_f32(v1_h))));

        int8_t *p_dst0 = dst + w * ALIGNED_CHANNEL_INT8;
        int8_t *p_dst1 = p_dst0 + ALIGNED_CHANNEL_INT8;

        vst1_s8(p_dst0, res8_0);
        vst1_s8(p_dst1, res8_1);

        const float *x0_ptr = (const float*)&x0;
        const float *x1_ptr = (const float*)&x1;
        for (int c = 8; c < 10; ++c) {
            int q0 = round_float_to_int(x0_ptr[c] * inv[c] + bias[c]);
            int q1 = round_float_to_int(x1_ptr[c] * inv[c] + bias[c]);
            if (q0 > 127) q0 = 127;
            if (q0 < -128) q0 = -128;
            if (q1 > 127) q1 = 127;
            if (q1 < -128) q1 = -128;
            p_dst0[c] = (int8_t)q0;
            p_dst1[c] = (int8_t)q1;
        }
    }

    for (; w < W; ++w) {
        const T &x = src[offset + w];
        float32x4_t v_l = vld1q_f32((const float*)&x);
        float32x4_t v_h = vld1q_f32((const float*)&x + 4);
        v_l = vfmaq_f32(bias_l, inv_l, v_l);
        v_h = vfmaq_f32(bias_h, inv_h, v_h);

        uint32x4_t m_l = vcltq_f32(v_l, v_zero);
        uint32x4_t m_h = vcltq_f32(v_h, v_zero);
        v_l = vaddq_f32(v_l, vbslq_f32(m_l, vnegq_f32(v_0_5), v_0_5));
        v_h = vaddq_f32(v_h, vbslq_f32(m_h, vnegq_f32(v_0_5), v_0_5));

        int8_t *p_dst = dst + w * ALIGNED_CHANNEL_INT8;
        int8x8_t res8 = vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_l)), vqmovn_s32(vcvtq_s32_f32(v_h))));
        vst1_s8(p_dst, res8);

        const float *x_ptr = (const float*)&x;
        for (int c = 8; c < 10; ++c) {
            int q = round_float_to_int(x_ptr[c] * inv[c] + bias[c]);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            p_dst[c] = (int8_t)q;
        }
    }

    return 0;
}

// asume N=H=1, C=8, C2=16, C1=1
template<typename T>
static inline int fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8(const boost::circular_buffer<T> &src, int8_t *dst, int W, 
                                    float scale, int zp, const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f) return -1;
    int offset = src.size() - W;
    if (offset < 0) return -1;

    // const param
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_0_5 = vdupq_n_f32(0.5f);
    int8_t zp_clipped = (int8_t)(zp > 127 ? 127 : (zp < -128 ? -128 : zp));
    memset(dst, zp_clipped, W * ALIGNED_CHANNEL_INT8);
    
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
        const T &x0 = src[offset + w];
        const T &x1 = src[offset + w + 1];

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
        const T &x = src[offset + w];
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

// asume N=H=1, C=8, C2=8, C1=1
template<typename T>
static inline int fast_NHWC_float32_circular_buffer_to_NC1HWC2_int16(const boost::circular_buffer<T> &src, int16_t *dst, int W, 
                                    float scale, int zp, const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f) return -1;
    int offset = src.size() - W;
    if (offset < 0) return -1;

    // const param
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_0_5 = vdupq_n_f32(0.5f);
    
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
        const T &x0 = src[offset + w];
        const T &x1 = src[offset + w + 1];

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
        int16x8_t res16_0 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v0_l)), vqmovn_s32(vcvtq_s32_f32(v0_h)));
        int16x8_t res16_1 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v1_l)), vqmovn_s32(vcvtq_s32_f32(v1_h)));

        // Store
        int16_t *p_dst = dst + w * ALIGNED_CHANNEL_INT16;

        vst1q_s16(p_dst, res16_0);      
        vst1q_s16(p_dst + 8, res16_1);
    }

    for (; w < W; ++w) {
        const T &x = src[offset + w];
        float32x4_t v_l = vld1q_f32((const float*)&x);
        float32x4_t v_h = vld1q_f32((const float*)&x + 4);
        v_l = vfmaq_f32(bias_l, inv_l, v_l);
        v_h = vfmaq_f32(bias_h, inv_h, v_h);
        
        uint32x4_t m_l = vcltq_f32(v_l, v_zero);
        uint32x4_t m_h = vcltq_f32(v_h, v_zero);
        v_l = vaddq_f32(v_l, vbslq_f32(m_l, vnegq_f32(v_0_5), v_0_5));
        v_h = vaddq_f32(v_h, vbslq_f32(m_h, vnegq_f32(v_0_5), v_0_5));

        int16x8_t res16 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_l)), vqmovn_s32(vcvtq_s32_f32(v_h)));
        vst1q_s16(dst + w * ALIGNED_CHANNEL_INT16, res16);
    }

    return 0;
}


// asume N=H=1, C=10, C2=16, C1=1
template<typename T>
static inline int fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8_C10(const boost::circular_buffer<T> &src, int8_t *dst, int W,
                                    float scale, int zp, const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f) return -1;
    int offset = src.size() - W;
    if (offset < 0) return -1;

    // const param
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_0_5 = vdupq_n_f32(0.5f);
    int8_t zp_clipped = (int8_t)(zp > 127 ? 127 : (zp < -128 ? -128 : zp));
    memset(dst, zp_clipped, W * ALIGNED_CHANNEL_INT8);

    float inv[10];
    float bias[10];
    for (int c = 0; c < 10; ++c) {
        float mc = (c < mean_std_len) ? mean[c] : 0.f;
        float sc = (c < mean_std_len && std[c] > 0.f) ? std[c] : 1.f;
        inv[c] = 1.0f / (sc * scale);
        bias[c] = (float)zp - mc * inv[c];
    }

    const float32x4_t inv_l = vld1q_f32(inv);
    const float32x4_t inv_h = vld1q_f32(inv + 4);
    const float32x4_t bias_l = vld1q_f32(bias);
    const float32x4_t bias_h = vld1q_f32(bias + 4);

    int w = 0;
    for (; w <= W - 2; w += 2) {
        const T &x0 = src[offset + w];
        const T &x1 = src[offset + w + 1];

        float32x4_t v0_l = vld1q_f32((const float*)&x0);
        float32x4_t v0_h = vld1q_f32((const float*)&x0 + 4);
        float32x4_t v1_l = vld1q_f32((const float*)&x1);
        float32x4_t v1_h = vld1q_f32((const float*)&x1 + 4);

        v0_l = vfmaq_f32(bias_l, inv_l, v0_l);
        v0_h = vfmaq_f32(bias_h, inv_h, v0_h);
        v1_l = vfmaq_f32(bias_l, inv_l, v1_l);
        v1_h = vfmaq_f32(bias_h, inv_h, v1_h);

        uint32x4_t m0_l = vcltq_f32(v0_l, v_zero);
        uint32x4_t m0_h = vcltq_f32(v0_h, v_zero);
        uint32x4_t m1_l = vcltq_f32(v1_l, v_zero);
        uint32x4_t m1_h = vcltq_f32(v1_h, v_zero);

        v0_l = vaddq_f32(v0_l, vbslq_f32(m0_l, vnegq_f32(v_0_5), v_0_5));
        v0_h = vaddq_f32(v0_h, vbslq_f32(m0_h, vnegq_f32(v_0_5), v_0_5));
        v1_l = vaddq_f32(v1_l, vbslq_f32(m1_l, vnegq_f32(v_0_5), v_0_5));
        v1_h = vaddq_f32(v1_h, vbslq_f32(m1_h, vnegq_f32(v_0_5), v_0_5));

        int8x8_t res8_0 = vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v0_l)), vqmovn_s32(vcvtq_s32_f32(v0_h))));
        int8x8_t res8_1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v1_l)), vqmovn_s32(vcvtq_s32_f32(v1_h))));

        int8_t *p_dst0 = dst + w * ALIGNED_CHANNEL_INT8;
        int8_t *p_dst1 = p_dst0 + ALIGNED_CHANNEL_INT8;

        vst1_s8(p_dst0, res8_0);
        vst1_s8(p_dst1, res8_1);

        const float *x0_ptr = (const float*)&x0;
        const float *x1_ptr = (const float*)&x1;
        for (int c = 8; c < 10; ++c) {
            int q0 = round_float_to_int(x0_ptr[c] * inv[c] + bias[c]);
            int q1 = round_float_to_int(x1_ptr[c] * inv[c] + bias[c]);
            if (q0 > 127) q0 = 127;
            if (q0 < -128) q0 = -128;
            if (q1 > 127) q1 = 127;
            if (q1 < -128) q1 = -128;
            p_dst0[c] = (int8_t)q0;
            p_dst1[c] = (int8_t)q1;
        }
    }

    for (; w < W; ++w) {
        const T &x = src[offset + w];
        float32x4_t v_l = vld1q_f32((const float*)&x);
        float32x4_t v_h = vld1q_f32((const float*)&x + 4);
        v_l = vfmaq_f32(bias_l, inv_l, v_l);
        v_h = vfmaq_f32(bias_h, inv_h, v_h);

        uint32x4_t m_l = vcltq_f32(v_l, v_zero);
        uint32x4_t m_h = vcltq_f32(v_h, v_zero);
        v_l = vaddq_f32(v_l, vbslq_f32(m_l, vnegq_f32(v_0_5), v_0_5));
        v_h = vaddq_f32(v_h, vbslq_f32(m_h, vnegq_f32(v_0_5), v_0_5));

        int8_t *p_dst = dst + w * ALIGNED_CHANNEL_INT8;
        int8x8_t res8 = vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_l)), vqmovn_s32(vcvtq_s32_f32(v_h))));
        vst1_s8(p_dst, res8);

        const float *x_ptr = (const float*)&x;
        for (int c = 8; c < 10; ++c) {
            int q = round_float_to_int(x_ptr[c] * inv[c] + bias[c]);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            p_dst[c] = (int8_t)q;
        }
    }

    return 0;
}
