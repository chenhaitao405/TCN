#pragma once
#include <chrono>
#include <cmath>
#include <data_stream.h>

template<typename T>
class FallDetector {
public:
    FallDetector()
        : g_value_(9.81f),
          weightless_factor_(0.7f),
          impact_factor_(1.5f),
          still_acc_lower_(0.7f * g_value_),
          still_acc_upper_(1.3f * g_value_),
          still_gyro_thresh_(0.35f),
          orientation_diff_thresh_deg_(60.0f),
          stage_(Stage::Idle),
          fall_detected_(false),
          initial_set_(false) {}

    // 每帧调用，返回true表示检测到一次跌倒事件
    bool update(const T& data) {
        using clock = std::chrono::steady_clock;
        const auto now = clock::now();

        const float acc_norm = norm3(&data.acc_x);
        const float gyro_norm = norm3(&data.gyro_x);

        // 采集初始站立姿态
        if (!initial_set_ && acc_norm > 0.5f * g_value_) {
            copy3(&data.acc_x, initial_orientation_);
            normalize3(initial_orientation_);
            initial_set_ = true;
        }

        switch (stage_) {
        case Stage::Idle:
            fall_detected_ = false;
            if (detectWeightless(acc_norm)) {
                stage_ = Stage::Weightless;
                t_weightless_ = now;
            }
            break;

        case Stage::Weightless: {
            const auto dt = now - t_weightless_;
            if (detectImpact(acc_norm) && dt < impact_window_) {
                stage_ = Stage::Impacted;
                t_impact_ = now;
            } else if (dt > max_weightless_span_) {
                reset();
            }
            break;
        }

        case Stage::Impacted: {
            const auto dt = now - t_impact_;
            if (detectStillness(acc_norm, gyro_norm) && dt < still_window_) {
                stage_ = Stage::Still;
                t_still_start_ = now;
            } else if (dt > still_window_) {
                reset();
            }
            break;
        }

        case Stage::Still: {
            const auto still_dt = now - t_still_start_;
            if (detectStillness(acc_norm, gyro_norm)) {
                if (still_dt >= min_still_duration_) {
                    if (orientationChanged(&data.acc_x)) {
                        fall_detected_ = true;
                        reset();
                        return true;
                    } else {
                        reset();
                    }
                }
            } else {
                reset();
            }
            break;
        }
        }
        return false;
    }

    void reset() {
        stage_ = Stage::Idle;
        t_weightless_ = std::chrono::steady_clock::now();
        t_impact_ = t_weightless_;
        t_still_start_ = t_weightless_;
    }

private:
    enum class Stage { Idle, Weightless, Impacted, Still };

    float g_value_;
    float weightless_factor_;
    float impact_factor_;
    float still_acc_lower_;
    float still_acc_upper_;
    float still_gyro_thresh_;
    float orientation_diff_thresh_deg_;

    Stage stage_;
    bool fall_detected_;
    bool initial_set_;
    float initial_orientation_[3] = {0.0f, 0.0f, 0.0f};

    std::chrono::steady_clock::time_point t_weightless_{};
    std::chrono::steady_clock::time_point t_impact_{};
    std::chrono::steady_clock::time_point t_still_start_{};

    // 时间窗口
    const std::chrono::milliseconds impact_window_{800};
    const std::chrono::milliseconds max_weightless_span_{1200};
    const std::chrono::milliseconds still_window_{2500};
    const std::chrono::milliseconds min_still_duration_{600};

    static float norm3(const float v[3]) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }

    static void copy3(const float src[3], float dst[3]) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
    }

    static void normalize3(float v[3]) {
        const float n = norm3(v);
        if (n > 1e-5f) {
            v[0] /= n;
            v[1] /= n;
            v[2] /= n;
        }
    }

    bool detectWeightless(float acc_norm) const {
        return acc_norm < weightless_factor_ * g_value_;
    }

    bool detectImpact(float acc_norm) const {
        return acc_norm > impact_factor_ * g_value_;
    }

    bool detectStillness(float acc_norm, float gyro_norm) const {
        return acc_norm > still_acc_lower_ && acc_norm < still_acc_upper_ && gyro_norm < still_gyro_thresh_;
    }

    bool orientationChanged(const float acc[3]) const {
        if (!initial_set_) return true; // 没有初始参考时，保持敏感
        float curr[3];
        copy3(acc, curr);
        normalize3(curr);
        float dot = initial_orientation_[0] * curr[0] + initial_orientation_[1] * curr[1] + initial_orientation_[2] * curr[2];
        dot = std::max(-1.0f, std::min(1.0f, dot));
        const float angle_deg = std::acos(dot) * 180.0f / 3.14159265f;
        return angle_deg > orientation_diff_thresh_deg_;
    }
};