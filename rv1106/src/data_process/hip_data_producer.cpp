#include "data_stream.h"
#include <memory>
#include "fall_detect/fall_detect.hpp"
#include "protocol/soc2mcu.h"
#include "data_process/hip_data_producer.h"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <vector>
#include <motor_pos_calib/motor_pos_calib.hpp>
namespace hip_producer {

Hip::DataStream data_stream;
std::unique_ptr<Lsm6dsr> lsm;
std::unique_ptr<SerialLoop<20>> serial2mcu;

static FallDetector<Hip::DataFrame> g_fall_detector;

bool start_serial2mcu(const std::string& portName) {
    std::vector<uint8_t> head = {HIP_DATA_HEAD1, HIP_DATA_HEAD2};
    std::vector<uint8_t> end = {HIP_DATA_END1, HIP_DATA_END2};
    timeval timeout = {0, 15000};  // 15ms超时
    
    // 初始化 IMU 传感器
    #ifdef DEBUG
        lsm = Lsm6dsr::create(I2C_BUS_ID, LSM_ADDR,
                              ACC_ODR_104 | ACC_4g,
                              GYRO_ODR_104 | GYRO_250dps,
                              true);
    #else 
        lsm = Lsm6dsr::create(I2C_BUS_ID, LSM_ADDR,
                              ACC_ODR_104 | ACC_4g,
                              GYRO_ODR_104 | GYRO_250dps,
                              false);
    #endif
    if (!lsm) {
        std::cerr << "[Error] Lsm6dsr initialization failed!" << std::endl;
        return false;
    }
    
    serial2mcu = SerialLoop<20>::create(portName, head, end, B115200, timeout);
    if (!serial2mcu) {
        lsm.reset();
        std::cerr << "[Error] serial2mcu initialization failed!" << std::endl;
        return false;
    }
    g_fall_detector.reset();
    serial2mcu->setCallback([&](const uint8_t* frame) {
        static MotorPosCalibate &motor_pos_calib = MotorPosCalibate::getInstance(); 
        ImuData imu_data;
        lsm->readImuData(imu_data);
        MotorData motor_data;
        memcpy(&motor_data, frame, sizeof(MotorData));
        motor_data.motorPosRight = -motor_data.motorPosRight;
        motor_data.motorVelRight = -motor_data.motorVelRight;
        if (!motor_pos_calib.is_calibrated()) {
            if (!motor_pos_calib.try_collect_zero_bias(motor_data.motorPosLeft, motor_data.motorPosRight)) {
                return;
            }
        }

        motor_pos_calib.apply_zero_bias(motor_data.motorPosLeft, motor_data.motorPosRight);
        #ifdef DEBUG
            static int count = 0;
            if (count++ % 20 == 0) {
                std::cout << std::fixed << std::setprecision(3)
                        << "Parsed Data -> "
                        << " | motorPosLeft: " << motor_data.motorPosLeft
                        << " | motorVelLeft: " << motor_data.motorVelLeft
                        << " | motorPosRight: " << motor_data.motorPosRight
                        << " | motorVelRight: " << motor_data.motorVelRight 
                        << " | acc_x: " << imu_data.acc_x
                        << " | acc_y: " << imu_data.acc_y
                        << " | acc_z: " << imu_data.acc_z
                        << " | gyro_x: " << imu_data.gyro_x
                        << " | gyro_y: " << imu_data.gyro_y
                        << " | gyro_z: " << imu_data.gyro_z << std::endl;
            }
        #endif

        Hip::DataFrame curr_frame;
        
        #ifndef HIP_MODEL_10_INPUT_CHANNEL
        if (Hip::ModelChannel == 8){
            curr_frame = Hip::DataFrame{imu_data.acc_x, imu_data.acc_y, imu_data.acc_z, imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z, motor_data.motorPosLeft, motor_data.motorPosRight};
        }
        #else
            curr_frame = Hip::DataFrame{imu_data.acc_x, imu_data.acc_y, imu_data.acc_z,
                imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z, motor_data.motorPosLeft, motor_data.motorPosRight, motor_data.motorVelLeft, motor_data.motorVelRight};
        #endif

        #ifdef FAIL_DETECT
            if (g_fall_detector.update(curr_frame)) {
                printf("[FallDetector] Potential fall detected\n");
            }
        #endif

        (void)data_stream.push(curr_frame);
    });

    serial2mcu->start();
    return true;
}

void sendData(float left_moment, uint8_t pattern, uint8_t fall_back) {
    if (!serial2mcu || !serial2mcu->isValid()) return;

    left_moment = std::max(-18.0f, std::min(18.0f, left_moment));
    
    uint8_t buffer_send[10] = {HIP_SEND_HEAD1, HIP_SEND_HEAD2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, HIP_SEND_END1, HIP_SEND_END2};
    memcpy(&buffer_send[2], &left_moment, sizeof(float));
    memcpy(&buffer_send[6], &pattern, sizeof(uint8_t));
    memcpy(&buffer_send[7], &fall_back, sizeof(uint8_t));
    serial2mcu->send(buffer_send, 10);
}

void stop_serial2mcu() {
    if (serial2mcu) {
        serial2mcu->stop();
        serial2mcu.reset();
    }
    if (lsm) {
        lsm->stop();
        lsm.reset();
    }
    g_fall_detector.reset();
}

} // namespace hip_producer