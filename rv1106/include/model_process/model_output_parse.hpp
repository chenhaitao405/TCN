#pragma once
#include <stdint.h>
#include "fp16/Float16.h"
#include <iostream>
#include <type_traits>
#include <rknn/rknn_api.h>

template <const int NumClass, typename T>
inline uint8_t get_pattern_from_output(const T *output_addr,
                                float logits[NumClass])
{
    if constexpr (std::is_void_v<T>)
    {
        throw "output should not be void* !\n";
    }
    float max_val = -1e9f;
    uint8_t pattern = 0x00;
    for (int i = 0; i < NumClass; ++i)
    {
        logits[i] = static_cast<float>(output_addr[i]);
        if (logits[i] > max_val)
        {
            max_val = logits[i];
            pattern = (uint8_t)i;
        }
    }
    return pattern;
}

inline float get_torque_from_output(void *output_addr, const rknn_tensor_attr &output_attr)
{
    int W, C2;
    if (output_attr.fmt == RKNN_TENSOR_NC1HWC2)
    {
        W = output_attr.dims[3];
        C2 = output_attr.dims[4];
    }
    else if (output_attr.fmt == RKNN_TENSOR_NCHW)
    {
        W = output_attr.dims[3];
        C2 = 1;
    }
    else if (output_attr.fmt == RKNN_TENSOR_NHWC)
    {
        W = output_attr.dims[2];
        C2 = output_attr.dims[3];
    }
    else
    {
        throw "output attr format not be supported!\n";
    }
    if (output_attr.type == RKNN_TENSOR_INT8)
    {
        int8_t *p = static_cast<int8_t *>(output_addr);
        return (static_cast<int>(p[(W - 1) * C2]) - output_attr.zp) * output_attr.scale;
    }
    else if (output_attr.type == RKNN_TENSOR_INT16)
    {
        int16_t *p = static_cast<int16_t *>(output_addr);
        return (static_cast<int>(p[(W - 1) * C2]) - output_attr.zp) * output_attr.scale;
    }
    else if (output_attr.type == RKNN_TENSOR_FLOAT16)
    {
        rknpu2::float16 *p = static_cast<rknpu2::float16 *>(output_addr);
        return (static_cast<float>(p[(W - 1) * C2]) - output_attr.zp) * output_attr.scale;
    }
    else if (output_attr.type == RKNN_TENSOR_FLOAT32)
    {
        float *p = static_cast<float *>(output_addr);
        return (static_cast<float>(p[(W - 1) * C2]) - output_attr.zp) * output_attr.scale;
    }
    else
    {
        throw "output data format not be supported!\n";
    }
}
