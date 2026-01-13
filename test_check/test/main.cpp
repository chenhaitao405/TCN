#include <arm_neon.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <memory.h>
#include "rknn_api.h"
#include <chrono>
#define ALIGNED_CHANNEL 8

void process_float_to_int8(std::vector<float> const &v, int8_t *dst) {
    // 1. 【四舍五入逻辑】
    // ARMv7 默认 vcvt 是向零取整 (Truncate)。
    // 算法：如果是正数加 0.5，如果是负数减 0.5，然后取整。
    
    float32x4_t input = vld1q_f32(v.data());
    // 生成一个全是 0 的向量，用于比较正负
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    // 生成全是 0.5 的向量
    float32x4_t v_0_5 = vdupq_n_f32(0.5f);
    
    // 比较 input < 0 ? (结果为全1掩码) : (结果为0)
    uint32x4_t mask_neg = vcltq_f32(input, v_zero);
    
    // 依据掩码选择偏移量：如果是负数选 -0.5，正数选 0.5
    // vbslq (Bitwise Select) 非常快，无分支逻辑
    float32x4_t v_offset = vbslq_f32(mask_neg, vnegq_f32(v_0_5), v_0_5);
    
    // 输入值 + 偏移量
    float32x4_t v_rounded = vaddq_f32(input, v_offset);
    
    // 转换浮点为 32位 有符号整数
    int32x4_t v_s32 = vcvtq_s32_f32(v_rounded);

    // 2. 【饱和窄化 (Saturating Narrowing)】
    // 第一步：int32x4 (128bit) -> int16x4 (64bit)
    // 此时范围被限制在 -32768 ~ 32767，但没关系，我们要的是 -128~127
    int16x4_t v_s16 = vqmovn_s32(v_s32);
    
    // 准备第二次窄化：vqmovn_s16 需要输入 128位的 int16x8
    // 我们只有 4 个数，所以用 vcombine 拼凑一个 dummy 向量（填充0即可）
    // 这样凑成：[v_s16 (低64位), 0 (高64位)]
    int16x8_t v_s16_comb = vcombine_s16(v_s16, vcreate_s16(0));
    
    // 第二步：int16x8 (128bit) -> int8x8 (64bit)
    // 关键点！vqmovn 会自动处理饱和。
    // 如果数值 > 127，它变成 127；如果 < -128，它变成 -128。
    int8x8_t v_s8 = vqmovn_s16(v_s16_comb);

    // 3. 【存储】
    // v_s8 现在有 8 个字节，前 4 个是我们计算的结果，后 4 个是刚才凑数的垃圾数据。
    // 我们只需要把前 4 个字节写入 dst。
    // 技巧：将 int8x8 视作 uint32x2，然后只写出第一个 Lane (也就是前4个字节)
    vst1_lane_u32((uint32_t*)dst, vreinterpret_u32_s8(v_s8), 0);
}

static inline int fast_NHWC_float32_deque_to_NC1HWC2_int8(float const *src, int8_t *dst,  int W, 
                                    float scale, int zp, const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f) return -1;

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
        float const * x0 = src + w * 8;
        float const * x1 = src + w * 8 + 8;

        float32x4_t v0_l = vld1q_f32(x0);
        float32x4_t v0_h = vld1q_f32(x0 + 4);

        float32x4_t v1_l = vld1q_f32(x1);
        float32x4_t v1_h = vld1q_f32(x1 + 4);

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
        float const *x = src + w * 8;
        float32x4_t v_l = vld1q_f32(x);
        float32x4_t v_h = vld1q_f32(x + 4);
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

static inline int fast_NHWC_float32_deque_to_NC1HWC2_int8_2(float const *src, int8_t *dst,  int W, 
                                    float scale, int zp, const float *mean, const float *std, int mean_std_len)
{
    if (scale <= 0.f) return -1;

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
    for (; w < W; ++w) {
        // load data
        float const * x0 = src + w * 8;

        float32x4_t v0_l = vld1q_f32(x0);
        float32x4_t v0_h = vld1q_f32(x0 + 4);

        // 归一化计算
        v0_l = vfmaq_f32(bias_l, inv_l, v0_l);
        v0_h = vfmaq_f32(bias_h, inv_h, v0_h);


        // Rounding
        uint32x4_t m0_l = vcltq_f32(v0_l, v_zero);
        uint32x4_t m0_h = vcltq_f32(v0_h, v_zero);

        v0_l = vaddq_f32(v0_l, vbslq_f32(m0_l, vnegq_f32(v_0_5), v_0_5));
        v0_h = vaddq_f32(v0_h, vbslq_f32(m0_h, vnegq_f32(v_0_5), v_0_5));

        // Saturating Narrowing
        int16x8_t s16_vec8_0 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v0_l)), vqmovn_s32(vcvtq_s32_f32(v0_h)));
        int8x8_t res8_0 = vqmovn_s16(s16_vec8_0);


        // Store
        int8_t *p_dst = dst + w * 16;

        vst1_s8(p_dst, res8_0);
    }

    for (; w < W; ++w) {
        float const *x = src + w * 8;
        float32x4_t v_l = vld1q_f32(x);
        float32x4_t v_h = vld1q_f32(x + 4);
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

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "missing model path\n";
        return 1;
    }

    std::cout << "请输入4个浮点数:\n";
    std::vector<float> v(4);
    for (float &num : v) {
        std::cin >> num;
    }
    int8_t result[4];
    process_float_to_int8(v, result);
    for (int i = 0; i < 4; ++i) {
        std::cout << (int)result[i] << std::endl;
    }
    char *model_path = argv[1];
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_path, 0, 0, NULL);
    if (ret != RKNN_SUCC)
    {
        printf("rknn init failed! ret=%d\n", ret);
        return -1;
    }
    rknn_tensor_attr input_attrs[1], output_attrs[1];
    input_attrs[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &input_attrs[0], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }

    output_attrs[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &output_attrs[0], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }

    rknn_tensor_mem *input_mems[1];
    input_attrs[0].pass_through = 1;
    
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    std::cout << "first:\n";
    std::cout << input_mems[0]->fd << std::endl << input_mems[0]->flags << std::endl 
            << input_mems[0]->offset << std::endl << input_mems[0]->size << std::endl
            << input_mems[0]->virt_addr << std::endl << input_mems[0]->phys_addr << std::endl;
    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    if (ret != RKNN_SUCC) {
        printf("rknn_set_io_mem input fail! ret=%d\n", ret);
        return -1;
    }
    std::cout << "second:\n";
    std::cout << input_mems[0]->fd << std::endl << input_mems[0]->flags << std::endl 
            << input_mems[0]->offset << std::endl << input_mems[0]->size << std::endl
            << input_mems[0]->virt_addr << std::endl << input_mems[0]->phys_addr << std::endl;
    rknn_destroy_mem(ctx, input_mems[0]);
    rknn_destroy(ctx);

    float mean[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float std[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float scale = 1.f;
    int zp = 0;
    int W = 280;
    float *src = (float*)aligned_alloc(16, W*8*sizeof(float));

    int8_t *dst = (int8_t*) aligned_alloc(16, W*ALIGNED_CHANNEL);
    memset(dst, 0, W*ALIGNED_CHANNEL);
    int8_t *dst2 = (int8_t*) aligned_alloc(16, W*ALIGNED_CHANNEL);
    memset(dst2, 0, W*ALIGNED_CHANNEL);
    // warmup
    for (int i=0; i<3; ++i) {
        fast_NHWC_float32_deque_to_NC1HWC2_int8(src, dst, W, scale, zp, mean, std, 8);
    }
    auto start_time1 = std::chrono::steady_clock::now();
    for (int i=0; i<500; ++i) {
        fast_NHWC_float32_deque_to_NC1HWC2_int8(src, dst, W, scale, zp, mean, std, 8);
    }
    auto end_time1 = std::chrono::steady_clock::now();
    auto during_time1 = std::chrono::duration_cast<std::chrono::microseconds>(end_time1 - start_time1).count();
    std::cout << during_time1 << "us" << std::endl;

    for (int i=0; i<3; ++i) {
        fast_NHWC_float32_deque_to_NC1HWC2_int8_2(src, dst2, W, scale, zp, mean, std, 8);
    }
    auto start_time2 = std::chrono::steady_clock::now();
    for (int i=0; i<500; ++i) {
        fast_NHWC_float32_deque_to_NC1HWC2_int8_2(src, dst2, W, scale, zp, mean, std, 8);
    }
    auto end_time2 = std::chrono::steady_clock::now();
    auto during_time2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time2 - start_time2).count();
    std::cout << during_time2 << "us" << std::endl;

    for (int i=0; i<W*ALIGNED_CHANNEL; ++i) {
        if (dst2[i] != dst[i]) {
            std::cout << "result wrong\n";
            delete[] src;
            delete[] dst;
            delete[] dst2;
            return 1;
        }
    }
    std::cout << "result right\n";

    delete[] src;
    delete[] dst;
    delete[] dst2;
    return 0;
}