#pragma once
#include <vector>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <iostream>

struct ZeroBiasCalibState {
    std::atomic<bool> calibrated{false};
    float offset_left = 0.0f;
    float offset_right = 0.0f;
    std::vector<float> left_samples;
    std::vector<float> right_samples;
    int retry_log_cnt = 0;
    std::mutex mtx;
};

const static int kCalibSamples = 50;
const static float kCalibStabilityThreshold = 0.05f;

class MotorPosCalibate {
public:
    static MotorPosCalibate& getInstance() {
        static MotorPosCalibate instance;
        return instance;
    }
    MotorPosCalibate(const MotorPosCalibate&) = delete;
    MotorPosCalibate& operator=(const MotorPosCalibate&) = delete;
    bool try_collect_zero_bias(float motor_pos_left, float motor_pos_right){
        if (calibrated.load()) {
            return true;
        }

        left_samples.push_back(motor_pos_left);
        right_samples.push_back(motor_pos_right);

        if (left_samples.size() < kCalibSamples || right_samples.size() < kCalibSamples) {
            return false;
        }

        const auto [min_l_it, max_l_it] = std::minmax_element(left_samples.begin(), left_samples.end());
        const auto [min_r_it, max_r_it] = std::minmax_element(right_samples.begin(), right_samples.end());
        const float left_span = *max_l_it - *min_l_it;
        const float right_span = *max_r_it - *min_r_it;

        if (left_span < kCalibStabilityThreshold && right_span < kCalibStabilityThreshold) {
            const float sum_l = std::accumulate(left_samples.begin(), left_samples.end(), 0.0f);
            const float sum_r = std::accumulate(right_samples.begin(), right_samples.end(), 0.0f);
            offset_left = sum_l / static_cast<float>(left_samples.size());
            offset_right = sum_r / static_cast<float>(right_samples.size());
            calibrated.store(true);
            std::cout << "[Calibration] Success. offset_left=" << offset_left
                    << ", offset_right=" << offset_right << std::endl;
            return true;
        }

        left_samples.clear();
        right_samples.clear();
        if ((retry_log_cnt++ % 5) == 0) {
            std::cout << "[Calibration] High fluctuation detected, waiting for standing still..." << std::endl;
        }
        return false;
    }

    inline void apply_zero_bias(float& motor_pos_left, float& motor_pos_right) {
        motor_pos_left -= offset_left;
        motor_pos_right -= offset_right;
    }

    void reset_zero_bias() {
        calibrated.store(false);
        offset_left = 0.0f;
        offset_right = 0.0f;
        left_samples.clear();
        right_samples.clear();
        retry_log_cnt = 0;
    }

    inline bool is_calibrated() {
        return calibrated.load();
    }

private:
    std::atomic<bool> calibrated;
    float offset_left;
    float offset_right;
    std::vector<float> left_samples;
    std::vector<float> right_samples;
    int retry_log_cnt;
    MotorPosCalibate(): calibrated(false), offset_left(0.0f), offset_right(0.0f), retry_log_cnt(0) {}
    ~MotorPosCalibate() {reset_zero_bias();}
};