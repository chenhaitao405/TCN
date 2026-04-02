#pragma once
#include <cmath>
#include <stdint.h>

class NonLinearFilter{
public:
    enum : uint8_t{
        SetInputLimit = 0b10000000,
        SetOutputLimit = 0b01000000,
        SetPowerPos = 0b00100000,
        SetPowerNeg = 0b00010000,
        SetGainPos = 0b00001000,
        SetGainNeg = 0b00000100,
        SetAll = 0xFF
    };

    static NonLinearFilter& getInstance() {
        static NonLinearFilter instance;
        return instance;
    }

    float nonlinear_filter_output(float x) {
        // 输入限幅
        x = clampf(x, -_input_limit, _input_limit);

        // 归一化到 [-1, 1]
        const float x_norm = x / _input_limit;
        float output_norm = 0.0f;

        if (x >= 0.0f) {
            output_norm = std::pow(std::fabs(x_norm), _power_pos) * _gain_pos;
        } else {
            output_norm = -std::pow(std::fabs(x_norm), _power_neg) * _gain_neg;
        }

        // 反归一化并输出限幅
        const float output = output_norm * _input_limit;
        return clampf(output, -_output_limit, _output_limit);
    }

    void init(float input_limit, float output_limit, float power_pos, float power_neg, float gain_pos, float gain_neg, uint8_t set_mask) {
        if (set_mask == 0) return;
        if (set_mask & SetInputLimit) _input_limit = input_limit;
        if (set_mask & SetOutputLimit) _output_limit = output_limit;
        if (set_mask & SetPowerPos) _power_pos = power_pos;
        if (set_mask & SetPowerNeg) _power_neg = power_neg;
        if (set_mask & SetGainPos) _gain_pos = gain_pos;
        if (set_mask & SetGainNeg) _gain_neg = gain_neg;
    }

private:
    float _input_limit;
    float _output_limit;
    float _power_pos;
    float _power_neg;
    float _gain_pos;
    float _gain_neg;
    inline float clampf(float v, float lo, float hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }
    NonLinearFilter() : _input_limit(0.40f), _output_limit(0.60f), _power_pos(2.00f), _power_neg(2.50f), _gain_pos(3.00f), _gain_neg(0.80f) {

    }
};