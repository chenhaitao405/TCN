#pragma once
#include <stdint.h>
#include "fp16/Float16.h"

template <const int T>
uint8_t get_pattern_from_output(const rknpu2::float16 *out_fp16,
                                        float logits[T])
{
    float max_val = -1e9f;
    uint8_t pattern = 0x00;
    for (int i = 0; i < T; i++) {
        logits[i] = (float)out_fp16[i];
        if (logits[i] > max_val) {
            max_val = logits[i];
            pattern = (uint8_t)i;
        }
    }
    return pattern;
}

template <const int T>
uint8_t get_pattern_from_output(const float *out_fp16,
                                        float logits[T])
{
    float max_val = -1e9f;
    uint8_t pattern = 0x00;
    for (int i = 0; i < T; i++) {
        logits[i] = (float)out_fp16[i];
        if (logits[i] > max_val) {
            max_val = logits[i];
            pattern = (uint8_t)i;
        }
    }
    return pattern;
}