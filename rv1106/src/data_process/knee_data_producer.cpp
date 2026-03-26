#include "data_stream.h"
#include <memory>
#include "fall_detect/fall_detect.hpp"
#include "protocol/soc2mcu.h"
#include "data_process/knee_data_producer.h"
#include <iostream>
#include <iomanip>

namespace knee_producer {

Knee::DataStream data_stream;

Dsp::SimpleFilter<Dsp::Butterworth::LowPass<2>, 1> motorvel_filter;
bool filter_initialized = false;
Knee::DataFrame prev_data;
bool has_prev_data = false;
std::unique_ptr<SerialLoop<48>> serial2mcu;
static char leg_side = 'l';

void set_leg_side(char side) {
    leg_side = side;
}

static FallDetector<KneeData> g_fall_detector;

static inline void interpolate_frame(const Knee::DataFrame& prev, const Knee::DataFrame& curr, float t, Knee::DataFrame& out) {
    out.gyro_x = prev.gyro_x + t * (curr.gyro_x - prev.gyro_x);
    out.gyro_y = prev.gyro_y + t * (curr.gyro_y - prev.gyro_y);
    out.gyro_z = prev.gyro_z + t * (curr.gyro_z - prev.gyro_z);
    out.acc_x = prev.acc_x + t * (curr.acc_x - prev.acc_x);
    out.acc_y = prev.acc_y + t * (curr.acc_y - prev.acc_y);
    out.acc_z = prev.acc_z + t * (curr.acc_z - prev.acc_z);
    out.motorPos = prev.motorPos + t * (curr.motorPos - prev.motorPos);
    out.motorVel = prev.motorVel + t * (curr.motorVel - prev.motorVel);
}

inline void transform_frame(const KneeData& data, Knee::DataFrame& transformed_data) {
    // 左右腿处理，see devices/custom_data_loader.py
    if (leg_side == 'l') {
        transformed_data.acc_x = data.acc_y;
        transformed_data.acc_y = -data.acc_x;
        transformed_data.acc_z = -data.acc_z;
        transformed_data.gyro_x = -data.gyro_y;
        transformed_data.gyro_y = data.gyro_x;
        transformed_data.gyro_z = data.gyro_z;
        transformed_data.motorPos = data.motorPosL - 180.0f;
        transformed_data.motorVel = data.motorVel / 2.0f;
    } else {
        transformed_data.acc_x = -data.acc_y;
        transformed_data.acc_y = -data.acc_x;
        transformed_data.acc_z = -data.acc_z;
        transformed_data.gyro_x = -data.gyro_y;
        transformed_data.gyro_y = -data.gyro_x;
        transformed_data.gyro_z = -data.gyro_z;
        transformed_data.motorPos = 180.0f - data.motorPosL;
        transformed_data.motorVel = -data.motorVel / 2.0f;
    }
}

bool start_serial2mcu(const std::string& portName) {
    std::vector<uint8_t> head = {KNEE_DATA_HEAD1, KNEE_DATA_HEAD2};
    std::vector<uint8_t> end = {KNEE_DATA_END1, KNEE_DATA_END2};
    timeval timeout = {0, 15000};  // 15ms超时
    
    serial2mcu = SerialLoop<48>::create(portName, head, end, B115200, timeout);
    if (!serial2mcu) {
        return false;
    }
    if (!filter_initialized) {
        motorvel_filter.setup(2, 200.0, 10.0);
        filter_initialized = true;
    }
    has_prev_data = false;
    g_fall_detector.reset();

    serial2mcu->setCallback([&](const uint8_t* frame) {
        KneeData knee_data;
        memcpy(&knee_data, frame, sizeof(KneeData));

        #ifdef DEBUG
            static int count = 0;
            if (count++ % 10 == 0) { // 每10帧打印一次
                std::cout << std::fixed << std::setprecision(3)
                        << "Parsed Data -> "
                        << " | motorPosL: " << knee_data.motorPosL
                        << " | motorVel: " << knee_data.motorVel
                        << " | acc_x: " << knee_data.acc_x
                        << " | gyro_x: " << knee_data.gyro_x << std::endl;
            }
        #endif

        #ifdef FAIL_DETECT
            if (g_fall_detector.update(knee_data)) {
                printf("[FallDetector] Potential fall detected\n");
            }
        #endif

        Knee::DataFrame interp_frame, curr_frame;
        transform_frame(knee_data, curr_frame);
        if (has_prev_data) {
            interpolate_frame(prev_data, curr_frame, 0.5f, interp_frame);
            #ifdef FILTER_INPUT
                float samples[2] = {interp_frame.motorVel, curr_frame.motorVel};
                float* channels[1] = {samples};
                motorvel_filter.process(2, channels);
                interp_frame.motorVel = samples[0];
                curr_frame.motorVel = samples[1];
            #endif

            data_stream.push(interp_frame);
            data_stream.push(curr_frame);
        } else {
            #ifdef FILTER_INPUT
                float* ch_curr[1] = {&curr_frame.motorVel};
                motorvel_filter.process(1, ch_curr);
            #endif
            data_stream.push(curr_frame);
        }
        
        prev_data = curr_frame;
        has_prev_data = true;
    });

    serial2mcu->start();
    return true;
}

void sendMoment(float left_moment, float time_val) {
    if (!serial2mcu || !serial2mcu->isValid()) return;

    left_moment = std::max(-18.0f, std::min(18.0f, left_moment));
    int8_t lef_m = static_cast<int8_t>(left_moment / 18.0f * 127);
    
    uint8_t buffer_send[6] = {KNEE_SEND_HEAD, 0x00, 0x00, 0x00, 0x00, KNEE_SEND_END}; // 示例协议
    buffer_send[1] = (uint8_t)lef_m;
    memcpy(&buffer_send[2], &time_val, 4);
    serial2mcu->send(buffer_send, 6);
}

void stop_serial2mcu() {
    if (serial2mcu) {
        serial2mcu->stop();
        serial2mcu.reset();
    }
    filter_initialized = false;
    has_prev_data = false;
    g_fall_detector.reset();
}

} // namespace knee_producer