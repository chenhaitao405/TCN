#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>
#include <csignal>
#include <memory>
#include <atomic>
#include <cassert>
#include <getopt.h>
#include "protocol/data_collection.h"
#include "protocol/soc2mcu.h"
#include "serial/uart_serial.h"
#include "model_process/model_process.h"
#include "boost/circular_buffer.hpp"
#include "data_stream.h"
#include "fp16/Float16.h"

static std::unique_ptr<SerialLoop<48>> g_driver;
static std::unique_ptr<SerialBase> wireless; 
std::atomic<bool> g_exit_flag{false};
bool PatternInfer = false;
bool TorqueInfer = false;

static void print_usage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " [--pattern] [--torque] [pattern_model] [torque_model] serial_mcu_port\n\n"
              << "Options:\n"
              << "  -p, --pattern    Enable pattern inference\n"
              << "  -t, --torque     Enable torque inference\n"
              << "  -h, --help       Show this help message\n\n"
              << "Examples:\n"
              << "  " << prog << " /dev/ttyS1\n"
              << "  " << prog << " --pattern knee_pattern.rknn /dev/ttyS1\n"
              << "  " << prog << " --torque knee_torque.rknn /dev/ttyS1\n";
}

static void send_moment(float left_moment, float time_val) {
    if (!g_driver || !g_driver->isValid()) return;

    left_moment = std::max(-18.0f, std::min(18.0f, left_moment));
    int8_t lef_m = static_cast<int8_t>(left_moment / 18.0f * 127);

    uint8_t buffer_send[6] = {KNEE_SEND_HEAD, 0x00, 0x00, 0x00, 0x00, KNEE_SEND_END};
    buffer_send[1] = (uint8_t)lef_m;
    memcpy(&buffer_send[2], &time_val, 4);
    g_driver->send(buffer_send, 6);
}

static void send_knee_frame(const KneeData& knee_data) {
    if (!wireless || !wireless->isValid()) return;

    uint8_t frame[49];
    frame[0] = KNEE_SEND_DATA_HEAD1;
    frame[1] = KNEE_SEND_DATA_HEAD2;
    memcpy(frame + 2, &knee_data, sizeof(KneeData));
    frame[46] = KNEE_SEND_DATA_END1;
    frame[47] = KNEE_SEND_DATA_END2;

    wireless->send(frame, sizeof(frame));
}

// 启停驱动接口
bool start_driver(const std::string& portName,
                  rknn_context* pattern_ctx,
                  std::vector<rknn_tensor_mem *> &pattern_input_mems,
                  std::vector<rknn_tensor_mem *> &pattern_output_mems,
                  std::vector<rknn_tensor_attr> const &pattern_input_attrs,
                  std::vector<rknn_tensor_attr> const &pattern_output_attrs,
                  rknn_context* torque_ctx,
                  std::vector<rknn_tensor_mem *> &torque_input_mems,
                  std::vector<rknn_tensor_mem *> &torque_output_mems,
                  std::vector<rknn_tensor_attr> const &torque_input_attrs,
                  std::vector<rknn_tensor_attr> const &torque_output_attrs) {
    std::vector<uint8_t> head = {KNEE_DATA_HEAD1, KNEE_DATA_HEAD2};
    std::vector<uint8_t> end = {KNEE_DATA_END1, KNEE_DATA_END2};
    timeval timeout = {0, 15000};  // 15ms超时
    
    g_driver = SerialLoop<48>::create(portName, head, end, B115200, timeout);
    if (!g_driver) {
        return false;
    }

    boost::circular_buffer<Knee::DataFrame> data_buffer(Knee::STREAM_LENGTH);

    g_driver->setCallback([&data_buffer,
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
        KneeData knee_data;
        memcpy(&knee_data, frame, sizeof(KneeData));

        uint8_t pattern = 0x00;
        float torque_output = knee_data.momentRev;
        if (PatternInfer || TorqueInfer) {
            data_buffer.push_back(Knee::DataFrame{
                knee_data.gyro_x,
                knee_data.gyro_y,
                knee_data.gyro_z,
                knee_data.acc_x,
                knee_data.acc_y,
                knee_data.acc_z,
                knee_data.motorPosL,
                knee_data.motorVel
            });

            if (data_buffer.full()) {
                if (PatternInfer && pattern_ctx && !pattern_input_mems.empty() && !pattern_output_mems.empty()) {
                    int ret = fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8<Knee::DataFrame>(
                        data_buffer,
                        (int8_t *)pattern_input_mems[0]->virt_addr,
                        Knee::STREAM_LENGTH,
                        pattern_input_attrs[0].scale,
                        pattern_input_attrs[0].zp,
                        Knee::g_mean_vals,
                        Knee::g_std_vals,
                        Knee::ModelChannel);
                    if (ret >= 0 && rknn_run(*pattern_ctx, NULL) == RKNN_SUCC) {
                        float logits[Knee::NumClasses];
                        rknpu2::float16 *out_fp16 = reinterpret_cast<rknpu2::float16 *>(pattern_output_mems[0]->virt_addr);
                        pattern = get_pattern_from_output<Knee::NumClasses>(out_fp16, logits);
                    }
                }

                if (TorqueInfer && torque_ctx && !torque_input_mems.empty() && !torque_output_mems.empty()) {
                    int ret = fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8(
                        data_buffer,
                        (int8_t *)torque_input_mems[0]->virt_addr,
                        Knee::STREAM_LENGTH,
                        torque_input_attrs[0].scale,
                        torque_input_attrs[0].zp,
                        Knee::g_mean_vals,
                        Knee::g_std_vals,
                        8);
                    if (ret >= 0 && rknn_run(*torque_ctx, NULL) == RKNN_SUCC) {
                        torque_output = (((int32_t)((int8_t *)torque_output_mems[0]->virt_addr)[Knee::STREAM_LENGTH - 1]) - torque_output_attrs[0].zp) * torque_output_attrs[0].scale;
                        knee_data.momentRev = torque_output;
                        send_moment(torque_output, 0.0f);
                    }
                }
            }
        }
        // KneeData 没有独立 pattern 字段，这里复用 time2 透传 pattern 结果。
        knee_data.time2 = static_cast<float>(pattern);

        #ifdef DEBUG
            static int count = 0;
            if (count++ % 100 == 0) { // 每10帧打印一次
                std::cout << std::fixed << std::setprecision(3)
                        << "Parsed Data -> "
                        << " | motorPosL: " << knee_data.motorPosL
                        << " | motorVel: " << knee_data.motorVel
                        << " | acc_x: " << knee_data.acc_x
                        << " | gyro_x: " << knee_data.gyro_x
                        << " | pattern: " << static_cast<int>(pattern)
                        << " | moment: " << torque_output << std::endl;
            }
        #endif

        send_knee_frame(knee_data);
    });

    g_driver->start();
    return true;
}

void stop_driver() {
    if (g_driver) {
        g_driver->stop();
        g_driver.reset();
    }
    wireless.reset();
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
    std::string serial_mcu_port;

    int opt;
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

    wireless = SerialBase::create("/dev/ttyS3", B921600);
    if (!wireless) {
        printf("start wireless serial fail!\n");
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
        assert(Knee::checkModelInfo(pattern_input_attrs));
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
        assert(Knee::checkModelInfo(torque_input_attrs));
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
    while (!g_exit_flag) {
        pause();
    }
    
    printf("Exiting gracefully...\n");
    stop_driver();
    resource_destroy(pattern_ctx, pattern_input_mems, pattern_output_mems);
    resource_destroy(torque_ctx, torque_input_mems, torque_output_mems);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
}