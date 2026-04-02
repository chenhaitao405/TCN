#include "rknn/rknn_api.h"
#include "model_process/model_process.h"
#include "data_process/hip_data_producer.h"
#include "timing/periodic_timer.h"
#include <cassert>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <string>
#include <thread>
#include <vector>
#include <boost/circular_buffer.hpp>
#include <algorithm>
#include "DspFilters/Butterworth.h"

static Dsp::SimpleFilter<Dsp::Butterworth::LowPass<2>, 1> g_output_filter;
static bool g_output_filter_initialized = false;

static inline float filter_output(float value) {
    float* channels[1] = {&value};
    g_output_filter.process(1, channels);
    return value;
}

constexpr int RECORD_FRAME_NUMS_DEFAULT = 5000;  // 录制帧数
constexpr int INFER_INTERVAL_MS = 5;   // 推理间隔 (ms)--200Hz
constexpr int FILL_CHECK_INTERVAL_MS = 50;
constexpr const char* RECORD_FILE_PATH_DEFAULT = "/data/hip_record_data.bin";  // 录制文件路径


static std::atomic<bool> g_running{true};

static void signal_handler(int signum) {
    printf("\nReceived signal %d, stopping...\n", signum);
    g_running = false;
}

void print_usage(const char* prog_name) {
    printf("Usage: %s [options] <model_path> <serial_port>\n\n", prog_name);
    printf("Description:\n");
    printf("    Run hip torque regression model inference or record raw sensor data.\n\n");
    printf("Positional Arguments:\n");
    printf("    model_path          Path to the RKNN model file\n");
    printf("    serial_port         Serial port for MCU communication (e.g., /dev/ttyS3)\n\n");
    printf("Options:\n");
    printf("    -s, --side <l|r>    Leg side: 'l' for left, 'r' for right (default: l)\n");
    printf("    -w, --weight <kg>   Body weight in kg for torque scaling (default: 70.0)\n");
    printf("    -i, --infer         Enable inference mode\n");
    printf("    -r, --record        Enable recording mode (save raw IMU data)\n");
    printf("    -p, --path <path>   Path to save recorded data (default: %s)\n", RECORD_FILE_PATH_DEFAULT);
    printf("    -n, --num <count>   Number of frames to record (default: %d)\n", RECORD_FRAME_NUMS_DEFAULT);
    printf("    -m, --moment        Enable torque transfer to exoskeleton\n");
    printf("    -d, --debug         Debug to show more information\n");
    printf("    -h, --help          Show this help message and exit\n\n");
    printf("Examples:\n");
    printf("    # Run torque inference on left leg with torque transfer\n");
    printf("    %s -i -m -s l model.rknn /dev/ttyS3\n\n", prog_name);
    printf("    # Record 5000 frames to custom path\n");
    printf("    %s -r -p /tmp/data.bin -n 5000 model.rknn /dev/ttyS3\n\n", prog_name);
    printf("    # Inference and record simultaneously\n");
    printf("    %s -i -r model.rknn /dev/ttyS3\n\n", prog_name);
}

int main(int argc, char** argv) {
    // 命令行选项定义
    static const struct option long_options[] = {
        {"side",     required_argument, nullptr, 's'},
        {"weight",   required_argument, nullptr, 'w'},
        {"infer",    no_argument,       nullptr, 'i'},
        {"record",   no_argument,       nullptr, 'r'},
        {"path",     required_argument, nullptr, 'p'},
        {"num",      required_argument, nullptr, 'n'},
        {"moment",   no_argument,       nullptr, 'm'},
        {"debug",    no_argument,       nullptr, 'd'},
        {"help",     no_argument,       nullptr, 'h'},
        {0,          0,                 0,        0 }
    };

    // 解析选项
    char side = 'l';
    float body_weight_kg = 70.0f;
    bool do_infer = false;
    bool record_data = false;
    bool debug = false;
    bool transfer_moment = false;
    const char* save_path = nullptr;
    int record_num = -1;
    int opt;

    while ((opt = getopt_long(argc, argv, "s:w:irp:n:mdh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 's':
                if (optarg[0] != 'l' && optarg[0] != 'r') {
                    printf("leg side is assigned wrong! get %s, but require 'r' or 'l'\n", optarg);
                    return 1;
                }
                side = optarg[0];
                printf("Processing for %s leg\n", side == 'l' ? "LEFT" : "RIGHT");
                break;

            case 'w': {
                char* end_ptr = nullptr;
                double parsed = std::strtod(optarg, &end_ptr);
                if (end_ptr == optarg || (end_ptr && *end_ptr != '\0') || parsed <= 0.0) {
                    printf("invalid body weight: %s\n", optarg);
                    return 1;
                }
                body_weight_kg = static_cast<float>(parsed);
                printf("Body weight set to %.1f kg\n", body_weight_kg);
                break;
            }

            case 'r':
                printf("record function is on\n");
                record_data = true;
                break;

            case 'p':
                save_path = optarg;
                printf("record save path is %s\n", save_path);
                break;

            case 'i':
                printf("infer function is on\n");
                do_infer = true;
                break;
            
            case 'm':
                printf("moment transfer function is on\n");
                transfer_moment = true;
                break;
            
            case 'd':
                printf("debug is on\n");
                debug = true;
                break;
            
            case 'n':
                record_num = atoi(optarg);
                if (record_num <= 0) {
                    printf("record num should be greater than 0, but got %d\n", record_num);
                    return -1;
                }
                printf("record num is %d\n", record_num);
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

    if (!do_infer && !record_data) {
        printf("Warning: No function specified (use -i for inference or -r for recording)\n");
    }

    if (optind + 2 > argc) {
        printf("Error: Missing required arguments (model_path serial_port)!\n");
        printf("Use '%s --help' for usage information.\n", argv[0]);
        return 1;
    }

    if (!save_path) save_path = RECORD_FILE_PATH_DEFAULT;
    if (record_num <= 0) record_num = RECORD_FRAME_NUMS_DEFAULT;

    const char* model_path = argv[optind];
    const std::string serial_port = argv[optind + 1];

    printf("\n=== Configuration ===\n");
    printf("Model path: %s\n", model_path);
    printf("Serial port: %s\n", serial_port.c_str());
    printf("Leg side: %s\n", side == 'l' ? "LEFT" : "RIGHT");
    printf("Body weight: %.1f kg\n", body_weight_kg);
    printf("Inference: %s\n", do_infer ? "enabled" : "disabled");
    printf("Recording: %s\n", record_data ? "enabled" : "disabled");
    if (record_data) {
        printf("\tSave path: %s\n", save_path);
        printf("\tMax frames: %d\n", record_num);
    }
    printf("Moment transfer: %s\n", transfer_moment ? "enabled" : "disabled");
    printf("Debug mode: %s\n", debug ? "enabled" : "disabled");
    printf("=====================\n\n");
    
    // hip_producer::set_leg_side(side);

    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 启动串口通信
    if (!hip_producer::start_serial2mcu(serial_port)) {
        printf("start serial2mcu failed\n");
        return -1;
    }

    // 模型初始化
    rknn_context ctx;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int ret = model_info_parse(ctx, const_cast<char*>(model_path), input_attrs, output_attrs);

    if (ret != RKNN_SUCC) {
        printf("model info parse failed!\n");
        hip_producer::stop_serial2mcu();
        return -1;
    }
    
    input_attrs[0].pass_through = 1;

    assert(Hip::checkModelInfo(input_attrs));
    assert(hip_producer::data_stream.is_lock_free());

    std::vector<rknn_tensor_mem *> input_mems;
    std::vector<rknn_tensor_mem *> output_mems;
    
    if (do_infer) {
        ret = alllocate_set_io_memory(ctx, input_attrs, output_attrs, input_mems, output_mems);
        if (ret != RKNN_SUCC) {
            printf("alllocate_set_io_memory fail! ret=%d\n", ret);
            resource_destroy(ctx, input_mems, output_mems);
            hip_producer::stop_serial2mcu();
            return -1;
        }
    }

    // open record file
    std::ofstream record_file;
    int recorded_count = 0;
    if (record_data) {
        record_file.open(save_path, std::ios::binary);
        if (!record_file.is_open()) {
            printf("Failed to open record file: %s\n", save_path);
            resource_destroy(ctx, input_mems, output_mems);
            hip_producer::stop_serial2mcu();
            return -1;
        }
        printf("Recording to: %s\n", save_path);
        printf("Data shape: channel=%d, length=%d\n", Hip::ModelChannel, Hip::STREAM_LENGTH);
    }

    // wait for data stream full
    timing::PeriodicTimer fill_timer;
    if (!fill_timer.start(FILL_CHECK_INTERVAL_MS)) {
        printf("create fill timerfd failed\n");
        if (record_data && record_file.is_open()) record_file.close();
        resource_destroy(ctx, input_mems, output_mems);
        hip_producer::stop_serial2mcu();
        return -1;
    }

    boost::circular_buffer<Hip::DataFrame> src_buffer(Hip::STREAM_LENGTH);
    while (g_running) {
        int cur = (int)hip_producer::data_stream.read_available();
        if (cur > 0) {
            hip_producer::data_stream.consume_all([&src_buffer](Hip::DataFrame frame)
                              { src_buffer.push_back(frame); });
        }
        printf("waiting for data stream full [%d/%d]\n", src_buffer.size(), Hip::STREAM_LENGTH);
        if (src_buffer.full())
            break;
        uint64_t missed = 0;
        if (!fill_timer.wait(missed, &g_running)) {
            printf("fill timer wait failed\n");
            if (record_data && record_file.is_open()) record_file.close();
            resource_destroy(ctx, input_mems, output_mems);
            hip_producer::stop_serial2mcu();
            return -1;
        }
    }
    fill_timer.stop();

    if (!g_running) {
        printf("interrupted before data ready\n");
        hip_producer::stop_serial2mcu();
        resource_destroy(ctx, input_mems, output_mems);
        return 0;
    }

    printf("data stream is full, ready to infer\n");
    
    if (do_infer && !g_output_filter_initialized) {
        g_output_filter.setup(2, 200.0, 5.0);
        g_output_filter_initialized = true;
    }
    int infer_count = 0;
    double total_infer_ms = 0.0;
    int moment_send_count = 0;

    timing::PeriodicTimer loop_timer;
    if (!loop_timer.start(INFER_INTERVAL_MS, 1)) {
        printf("create infer timerfd failed\n");
        if (record_data && record_file.is_open()) record_file.close();
        resource_destroy(ctx, input_mems, output_mems);
        hip_producer::stop_serial2mcu();
        return -1;
    }
    
    printf("Starting %s loop (every %dms), press Ctrl+C to stop...\n", 
           do_infer ? (record_data ? "inference+recording" : "inference") : "recording", 
           INFER_INTERVAL_MS);
    
    
    while (g_running) {
        uint64_t missed = 0;
        if (!loop_timer.wait(missed, &g_running)) {
            printf("infer timer wait failed\n");
            break;
        }
        if (missed > 1) {
            printf("[Warning] Overrun! Missed %llu pulses\n", (unsigned long long)(missed - 1));
        }

        // 从数据流获取最新数据
        if (hip_producer::data_stream.read_available()) {
            hip_producer::data_stream.consume_all([&src_buffer](Hip::DataFrame frame) {
                src_buffer.push_back(frame);
            });
        }

        if (record_data) {
            std::pair<Hip::DataFrame *, std::size_t> a1 = src_buffer.array_one();
            std::pair<Hip::DataFrame *, std::size_t> a2 = src_buffer.array_two();
            if (a1.second > 0)
                record_file.write(reinterpret_cast<const char*>(a1.first), a1.second * sizeof(Hip::DataFrame));
            if (a2.second > 0)
                record_file.write(reinterpret_cast<const char*>(a2.first), a2.second * sizeof(Hip::DataFrame));
            recorded_count++;
            
            if (recorded_count % 100 == 0) {
                printf("[Recording] Saved %d frames\n", recorded_count);
            }
            
            if (!do_infer && recorded_count >= record_num) {
                printf("Reached max record frames (%d), stopping...\n", record_num);
                break;
            }
        
            if (!do_infer) continue;
        }

        ret = fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8<Hip::DataFrame>(
            src_buffer,
            (int8_t*)input_mems[0]->virt_addr,
            Hip::STREAM_LENGTH,
            input_attrs[0].scale,
            input_attrs[0].zp,
            Hip::g_mean_vals,
            Hip::g_std_vals,
            Hip::ModelChannel
        );
        if (ret < 0)
        {
            printf("data buffer is not full\n");
            continue;
        }

        // model infer
        auto infer_start = std::chrono::steady_clock::now();
        ret = rknn_run(ctx, NULL);
        auto infer_end = std::chrono::steady_clock::now();
        
        if (ret != RKNN_SUCC) {
            printf("rknn run error %d\n", ret);
            break;
        }

        double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        total_infer_ms += infer_ms;
        infer_count++;

        float raw_output = get_torque_from_output(output_mems[0]->virt_addr, output_attrs[0]);
        float filtered_output = filter_output(raw_output);

        moment_send_count++;
        if (transfer_moment && moment_send_count % 2 == 0)
            hip_producer::sendData(filtered_output * body_weight_kg * 0.2f, 0x00, 0x00);

        // 打印统计信息
        if (debug && infer_count % 20 == 0) {
            const auto& last = src_buffer.back();
            printf("[%5d] gyro=(%.2f, %.2f, %.2f) | acc=(%.2f, %.2f, %.2f)\n",
                   infer_count,
                   last.gyro_x, last.gyro_y, last.gyro_z,
                   last.acc_x, last.acc_y, last.acc_z);

            double avg_infer_ms = total_infer_ms / infer_count;
            printf("        => raw=%.4f | torque=%.4f | avg=%.2fms | %.1f fps\n",
                   raw_output, filtered_output, avg_infer_ms, 1000.0 / avg_infer_ms);
        }

        // 如果同时记录，保存扭矩输出
        if (record_data) {
            record_file.write(reinterpret_cast<const char*>(&raw_output), sizeof(float));
            record_file.write(reinterpret_cast<const char*>(&filtered_output), sizeof(float));
            if (recorded_count >= record_num) {
                printf("Reached max record frames (%d), stopping...\n", record_num);
                break;
            }
        }
    }
    loop_timer.stop();

    // 关闭记录文件
    if (record_data && record_file.is_open()) {
        record_file.close();
        printf("\n=== Recording Complete ===\n");
        printf("Total frames recorded: %d\n", recorded_count);
        printf("File saved to: %s\n", save_path);
        printf("Data size: %.2f MB\n", 
             do_infer ? (recorded_count * Hip::STREAM_LENGTH * sizeof(Hip::DataFrame) + recorded_count * (2 * sizeof(float))) / (1024.0 * 1024.0) :
             (recorded_count * Hip::STREAM_LENGTH * sizeof(Hip::DataFrame)) / (1024.0 * 1024.0));
    }

    // 打印最终统计
    if (do_infer && infer_count > 0) {
        printf("\n=== Final Statistics ===\n");
        printf("Total inferences: %d\n", infer_count);
        printf("Average inference time: %.3f ms\n", total_infer_ms / infer_count);
        printf("Average FPS: %.1f\n", 1000.0 / (total_infer_ms / infer_count));
    }

    hip_producer::sendData(0.0f, 0x00, 0x00);
    hip_producer::stop_serial2mcu();
    resource_destroy(ctx, input_mems, output_mems);
    
    printf("Program exited cleanly.\n");
    return 0;
}
