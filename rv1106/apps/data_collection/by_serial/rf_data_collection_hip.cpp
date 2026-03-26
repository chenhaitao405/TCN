#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>
#include <csignal>
#include <memory>
#include "protocol/data_collection.h"
#include "protocol/soc2mcu.h"
#include "serial/uart_serial.h"
#include "peripherals/lsm6dsr.h"
#include "model_process/model_process.h"
#include <getopt.h>
#include "boost/circular_buffer.hpp"
#include "data_stream.h"
#include "fp16/Float16.h"
#include <atomic>
#include <cassert>
#include <algorithm>
#include <numeric>
#include "motor_pos_calib/motor_pos_calib.hpp"
#include <fstream>

static std::unique_ptr<SerialLoop<20>> g_driver;
static std::unique_ptr<Lsm6dsr> lsm;
static std::unique_ptr<SerialBase> wireless; 
std::atomic<bool> g_exit_flag = false;
bool TorqueInfer = false;
bool PatternInfer = false;
static std::ofstream g_pattern_conf_csv;
static std::atomic<uint64_t> g_pattern_conf_seq{0};
static bool g_pattern_conf_enabled = true;

static bool init_pattern_conf_csv(const std::string& csv_path) {
    {
        std::ifstream existing(csv_path);
        if (existing.good()) {
            std::cerr << "[Warn] csv already exists, skip csv logging: " << csv_path << std::endl;
            g_pattern_conf_enabled = false;
            return true;
        }
    }

    g_pattern_conf_csv.open(csv_path, std::ios::out);
    if (!g_pattern_conf_csv.is_open()) {
        std::cerr << "[Error] failed to open pattern/confidence csv: " << csv_path << std::endl;
        return false;
    }

    g_pattern_conf_csv << "seq,timestamp_ms,"
                      << "acc_x,acc_y,acc_z,"
                      << "gyro_x,gyro_y,gyro_z,"
                      << "motorAngle_L,motorAngle_R,motorAngleVel_L,motorAngleVel_R,"
                      << "pattern,confidence\n";
    g_pattern_conf_csv.flush();
    return true;
}

static inline void log_pattern_confidence(const ImuData& imu_data,
                                          const MotorData& motor_data,
                                          uint8_t pattern,
                                          float confidence) {
    if (!g_pattern_conf_enabled || !g_pattern_conf_csv.is_open()) {
        return;
    }
    // Assumes callback write path is single-threaded.
    const uint64_t seq = g_pattern_conf_seq.fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::system_clock::now();
    const auto ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    g_pattern_conf_csv << seq << ',' << ts_ms << ','
                       << imu_data.acc_x << ',' << imu_data.acc_y << ',' << imu_data.acc_z << ','
                       << imu_data.gyro_x << ',' << imu_data.gyro_y << ',' << imu_data.gyro_z << ','
                       << motor_data.motorPosLeft << ',' << motor_data.motorPosRight << ','
                       << motor_data.motorVelLeft << ',' << motor_data.motorVelRight << ','
                       << static_cast<int>(pattern) << ',' << confidence << '\n';
    // Flush each row so sudden power loss does not leave data only in user-space buffers.
    g_pattern_conf_csv.flush();
}

static void print_usage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " [--pattern] [--torque] [pattern_model] [torque_model] serial_mcu_port\n\n"
              << "Options:\n"
              << "  -p, --pattern    Enable pattern inference\n"
              << "  -t, --torque     Enable torque inference\n"
              << "  -h, --help       Show this help message\n\n"
              << "Examples:\n"
              << "  " << prog << " /dev/ttyS1\n"
              << "  " << prog << " --pattern hip_pattern.rknn /dev/ttyS1\n"
              << "  " << prog << " --torque hip_torque.rknn /dev/ttyS1\n"
              << "  " << prog << " --pattern --torque hip_pattern.rknn hip_torque.rknn /dev/ttyS1\n";
}

uint8_t uart_Send_Sensor[53]={HIP_SEND_DATA_HEAD1,HIP_SEND_DATA_HEAD2,
                            0x00,0x01,0x02,0x03, // acc_x
                            0x00,0x01,0x02,0x03, // acc_y
                            0x00,0x01,0x02,0x03, // acc_z
                            0x00,0x01,0x02,0x03, // gyro_x
                            0x00,0x01,0x02,0x03, // gyro_y
                            0x00,0x01,0x02,0x03, // gyro_z
                            0x00,0x01,0x02,0x03, // motorAngle_L
                            0x00,0x01,0x02,0x03, // motorAngle_R
                            0x00,0x01,0x02,0x03, // motorAngleVel_L
                            0x00,0x01,0x02,0x03, // motorAngleVel_R
                            0x00,                // pattern_result
                            0x00,0x01,0x02,0x03, // Torque
                            0x00,0x01,0x02,0x03, // Confidence
                            HIP_SEND_DATA_END1,HIP_SEND_DATA_END2};


// 向mcu发送数据
void sendData(float left_moment, uint8_t pattern, uint8_t fall_back) {
    if (!g_driver || !g_driver->isValid()) return;

    left_moment = std::max(-18.0f, std::min(18.0f, left_moment));
    
    uint8_t buffer_send[10] = {HIP_SEND_HEAD1, HIP_SEND_HEAD2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, HIP_SEND_END1, HIP_SEND_END2};
    memcpy(&buffer_send[2], &left_moment, sizeof(float));
    memcpy(&buffer_send[6], &pattern, sizeof(uint8_t));
    memcpy(&buffer_send[7], &fall_back, sizeof(uint8_t));
    g_driver->send(buffer_send, 10);
}

// 启停驱动接口
bool start_driver(const std::string& portName,
                  rknn_context* pattern_ctx,
                  const std::vector<rknn_tensor_mem *>& pattern_input_mems,
                  const std::vector<rknn_tensor_mem *>& pattern_output_mems,
                  std::vector<rknn_tensor_attr> const &pattern_input_attrs,
                  std::vector<rknn_tensor_attr> const &pattern_output_attrs,
                  rknn_context* torque_ctx,
                  const std::vector<rknn_tensor_mem *>& torque_input_mems,
                  const std::vector<rknn_tensor_mem *>& torque_output_mems,
                  std::vector<rknn_tensor_attr> const &torque_input_attrs,
                  std::vector<rknn_tensor_attr> const &torque_output_attrs) {
    std::vector<uint8_t> head = {HIP_DATA_HEAD1, HIP_DATA_HEAD2};
    std::vector<uint8_t> end = {HIP_DATA_END1, HIP_DATA_END2};
    timeval timeout = {0, 15000};  // 15ms超时
    
    g_driver = SerialLoop<20>::create(portName, head, end, B115200, timeout);
    if (!g_driver) {
        return false;
    }
    std::shared_ptr<boost::circular_buffer<Hip::DataFrame>> data_buffer = std::make_shared<boost::circular_buffer<Hip::DataFrame>>(Hip::STREAM_LENGTH);
    g_driver->setCallback([data_buffer,
                           pattern_ctx,
                           &pattern_input_mems,
                           &pattern_output_mems,
                           &pattern_input_attrs,
                           &pattern_output_attrs,
                           torque_ctx,
                           &torque_input_mems,
                           &torque_output_mems,
                           &torque_input_attrs,
                           &torque_output_attrs](const uint8_t* frame) {
        ImuData imu_data;
        lsm->readImuData(imu_data);
        static MotorPosCalibate &motor_pos_calib = MotorPosCalibate::getInstance();
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
        memcpy(&uart_Send_Sensor[2], &imu_data.acc_x, sizeof(float)*6);
        memcpy(&uart_Send_Sensor[26], &motor_data.motorPosLeft, 4);
        memcpy(&uart_Send_Sensor[30], &motor_data.motorPosRight, 4);
        memcpy(&uart_Send_Sensor[34], &motor_data.motorVelLeft, 4);
        memcpy(&uart_Send_Sensor[38], &motor_data.motorVelRight, 4);
        uint8_t pattern = 0x00;
        float torque_output = 0.0f;
        float confidence = -1.0f;
        if (PatternInfer || TorqueInfer) {
            #ifndef HIP_MODEL_10_INPUT_CHANNEL
            if (Hip::ModelChannel == 8) 
                data_buffer->push_back(Hip::DataFrame{imu_data.acc_x, imu_data.acc_y, imu_data.acc_z,
                imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z, motor_data.motorPosLeft, motor_data.motorPosRight});
            #else
                data_buffer->push_back(Hip::DataFrame{imu_data.acc_x, imu_data.acc_y, imu_data.acc_z,
                imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z, motor_data.motorPosLeft, motor_data.motorPosRight, motor_data.motorVelLeft, motor_data.motorVelRight});
            #endif
            if (data_buffer->full()) {
                if (PatternInfer && pattern_ctx && !pattern_input_mems.empty() && !pattern_output_mems.empty()) {
                    int ret = fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8<Hip::DataFrame>(
                        *data_buffer,
                        (int8_t *)pattern_input_mems[0]->virt_addr,
                        Hip::STREAM_LENGTH,
                        pattern_input_attrs[0].scale,
                        pattern_input_attrs[0].zp,
                        Hip::g_mean_vals,
                        Hip::g_std_vals,
                        Hip::ModelChannel);
                    
                    if (ret >= 0 && rknn_run(*pattern_ctx, NULL) == RKNN_SUCC) {
                        float logits[Hip::NumClasses];
                        rknpu2::float16 *out_fp16 = static_cast<rknpu2::float16 *>(pattern_output_mems[0]->virt_addr);
                        pattern = get_pattern_from_output<Hip::NumClasses>(out_fp16, logits);
                        for (int i = 0; i < Hip::NumClasses; ++i) {
                            confidence = std::max(logits[i], confidence);
                        }
                    }
                }

                if (TorqueInfer && torque_ctx && !torque_input_mems.empty() && !torque_output_mems.empty()) {
                    int ret = fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8<Hip::DataFrame>(
                        *data_buffer,
                        (int8_t *)torque_input_mems[0]->virt_addr,
                        Hip::STREAM_LENGTH,
                        torque_input_attrs[0].scale,
                        torque_input_attrs[0].zp,
                        Hip::g_mean_vals,
                        Hip::g_std_vals,
                        Hip::ModelChannel);
                    if (ret >= 0 && rknn_run(*torque_ctx, NULL) == RKNN_SUCC) {
                        torque_output = (static_cast<int32_t>(((int8_t *)torque_output_mems[0]->virt_addr)[Hip::STREAM_LENGTH - 1]) - torque_output_attrs[0].zp) 
                                        * torque_output_attrs[0].scale;
                    }
                }
            }
        }
        uart_Send_Sensor[42] = pattern;
        memcpy(&uart_Send_Sensor[43], &torque_output, sizeof(float));
        memcpy(&uart_Send_Sensor[47], &confidence, sizeof(float));
        log_pattern_confidence(imu_data, motor_data, pattern, confidence);
        wireless->send(uart_Send_Sensor, sizeof(uart_Send_Sensor));
        #ifdef DEBUG
            static int count = 0;
            if (count++ % 20 == 0) { // 每20帧打印一次
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
                        << " | gyro_z: " << imu_data.gyro_z 
                        << " | pattern: "<< static_cast<int>(pattern) 
                        << " | torque: " << torque_output
                        << " | conf: " << confidence <<std::endl;
            }
        #endif
        sendData(torque_output, pattern, 0x00);
    });

    g_driver->start();
    return true;
}

static void stop(int signum) {
    printf("\nReceived signal %d, stopping...\n", signum);
    g_exit_flag = true;
}

int main(int argc, char **argv) {
    static const struct option long_options[] = {
        {"pattern", no_argument, nullptr, 'p'},
        {"torque", no_argument, nullptr, 't'},
        {"help", no_argument, nullptr, 'h'},
        {0, 0, 0, 0}};
    const char *pattern_model_path = nullptr;
    const char *torque_model_path = nullptr;
    int opt;
    std::string serial_mcu_port; 
    while ((opt = getopt_long(argc, argv, "pth", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'p':
            PatternInfer = true;
            break;
        
        case 't':
            TorqueInfer = true;
            break;

        case 'h':
            print_usage(argv[0]);
            return 0;

        case '?':
        default:
            printf("Error: Unknown option\n");
            print_usage(argv[0]);
            return 1;
        }
    }

    int required_args = 1 + (PatternInfer ? 1 : 0) + (TorqueInfer ? 1 : 0);
    if (optind + required_args > argc) {
        printf("Error: Missing required arguments. Need: [pattern_model] [torque_model] serial_mcu_port (depending on --pattern/--torque)!\n");
        print_usage(argv[0]);
        return 1;
    }
    if (PatternInfer) {
        pattern_model_path = argv[optind++];
    }
    if (TorqueInfer) {
        torque_model_path = argv[optind++];
    }
    serial_mcu_port = argv[optind++];

    if (!init_pattern_conf_csv("/root/pattern_confidence_log.csv")) {
        return -1;
    }

    wireless = SerialBase::create("/dev/ttyS5", B921600);
    if (!wireless) {
        std::cerr << "[Error] wireless initialization failed!" << std::endl;
        return -1;
    }
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
        return -1;
    }

    rknn_context pattern_ctx = 0;
    std::vector<rknn_tensor_attr> pattern_input_attrs;
    std::vector<rknn_tensor_attr> pattern_output_attrs;
    std::vector<rknn_tensor_mem *> pattern_input_mems;
    std::vector<rknn_tensor_mem *> pattern_output_mems;

    rknn_context torque_ctx = 0;
    std::vector<rknn_tensor_attr> torque_input_attrs;
    std::vector<rknn_tensor_attr> torque_output_attrs;
    std::vector<rknn_tensor_mem *> torque_input_mems;
    std::vector<rknn_tensor_mem *> torque_output_mems;

    if (PatternInfer) {
        int ret = model_info_parse(pattern_ctx, const_cast<char *>(pattern_model_path), pattern_input_attrs, pattern_output_attrs);
        if (ret != RKNN_SUCC) {
            std::cerr << "[Error] pattern model information parse failed!" << std::endl;
            return -1;
        }
        pattern_input_attrs[0].pass_through = 1;
        assert(pattern_input_attrs[0].fmt == RKNN_TENSOR_NC1HWC2);
        assert(Hip::checkModelInfo(pattern_input_attrs));
        ret = alllocate_set_io_memory(pattern_ctx, pattern_input_attrs, pattern_output_attrs, pattern_input_mems, pattern_output_mems);
        if (ret != RKNN_SUCC) {
            std::cerr << "[Error] alllocate and set pattern model input output memory failed!" << std::endl;
            resource_destroy(pattern_ctx, pattern_input_mems, pattern_output_mems);
            return -1;
        }
    }

    if (TorqueInfer) {
        int ret = model_info_parse(torque_ctx, const_cast<char *>(torque_model_path), torque_input_attrs, torque_output_attrs);
        if (ret != RKNN_SUCC) {
            std::cerr << "[Error] torque model information parse failed!" << std::endl;
            resource_destroy(pattern_ctx, pattern_input_mems, pattern_output_mems);
            return -1;
        }
        torque_input_attrs[0].pass_through = 1;
        assert(torque_input_attrs[0].fmt == RKNN_TENSOR_NC1HWC2);
        assert(Hip::checkModelInfo(torque_input_attrs));
        ret = alllocate_set_io_memory(torque_ctx, torque_input_attrs, torque_output_attrs, torque_input_mems, torque_output_mems);
        if (ret != RKNN_SUCC) {
            std::cerr << "[Error] alllocate and set torque model input output memory failed!" << std::endl;
            resource_destroy(pattern_ctx, pattern_input_mems, pattern_output_mems);
            resource_destroy(torque_ctx, torque_input_mems, torque_output_mems);
            return -1;
        }
    }

    if (!start_driver(serial_mcu_port,
                      PatternInfer ? &pattern_ctx : nullptr,
                      pattern_input_mems,
                      pattern_output_mems,
                      pattern_input_attrs,
                      pattern_output_attrs,
                      TorqueInfer ? &torque_ctx : nullptr,
                      torque_input_mems,
                      torque_output_mems,
                      torque_input_attrs,
                      torque_output_attrs)) {
        printf("start driver fail!\n");
        return -1;
    }
   
    signal(SIGINT, stop);
    signal(SIGTERM, stop);
    
    printf("Driver started. Press Ctrl+C to exit...\n");
    
    // 主循环等待退出信号
    while (!g_exit_flag.load()) {
        pause();
    }
    g_driver.reset();
    if (g_pattern_conf_csv.is_open()) {
        g_pattern_conf_csv.flush();
        g_pattern_conf_csv.close();
    }
    resource_destroy(pattern_ctx, pattern_input_mems, pattern_output_mems);
    resource_destroy(torque_ctx, torque_input_mems, torque_output_mems);
    printf("Exiting gracefully...\n");
    return 0;
}