#include "rknn_api.h"
#include "model_info_parse.hpp"
#include "data_stream.h"
#include "DspFilters/Butterworth.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <arm_neon.h>

// ================= 常量定义 =================
constexpr int RECORD_FRAME_NUMS_DEFAULT = 5000;
constexpr int INFER_INTERVAL_MS = 5;   // 推理间隔 (ms)--200Hz
constexpr const char* RECORD_FILE_PATH_DEFAULT = "/data/record_data.bin";

// ================= 外部接口声明 =================
extern bool start_imu_driver(const std::string& portName);
extern void stop_imu_driver();
extern DataStream& get_data_stream();
extern std::mutex& get_data_stream_mutex();
extern void set_imu_side(char side);
extern void send_moment(float moment, float time_val);

// ================= 全局变量 =================
static std::atomic<bool> g_running{true};

// 模型归一化参数：[gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, motorPos, motorVel]
static const float g_mean_vals[8] = {
    0.7113f, -0.4300f, 0.7254f,   // gyro
    2.6529f, 8.7143f, -0.2818f,   // acc
    -30.6663f, -0.1348f           // motor
};
static const float g_std_vals[8] = {
    21.3148f, 45.2281f, 81.4140f, // gyro
    3.9808f, 4.4334f, 1.8336f,    // acc
    27.8284f, 107.1712f           // motor
};

static void signal_handler(int signum) {
    printf("\nReceived signal %d, stopping...\n", signum);
    g_running = false;
}

static void resource_destroy(rknn_context ctx, rknn_tensor_mem* input_mem, rknn_tensor_mem* output_mem) {
    if (input_mem) rknn_destroy_mem(ctx, input_mem);
    if (output_mem) rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
}

// ================= 输出滤波器 =================
// 2阶巴特沃斯低通滤波器，截止频率10Hz，采样率100Hz（推理频率）
static Dsp::SimpleFilter<Dsp::Butterworth::LowPass<2>, 1> g_output_filter;
static bool g_output_filter_initialized = false;

static float filter_output(float value) {
    float* channels[1] = {&value};
    g_output_filter.process(1, channels);
    return value;
}

void print_usage(const char* prog_name) {
    printf("Usage: %s [options] <model_path> <imu_port>\n\n", prog_name);
    printf("Description:\n");
    printf("    Run TCN model inference on IMU data or record raw sensor data.\n\n");
    printf("Positional Arguments:\n");
    printf("    model_path          Path to the RKNN model file\n");
    printf("    imu_port            Serial port for IMU device (e.g., /dev/ttyUSB0)\n\n");
    printf("Options:\n");
    printf("    -s, --side <l|r>    Leg side: 'l' for left, 'r' for right (default: l)\n");
    printf("    -i, --infer         Enable inference mode\n");
    printf("    -r, --record        Enable recording mode (save raw IMU data)\n");
    printf("    -p, --path <path>   Path to save recorded data (default: %s)\n", RECORD_FILE_PATH_DEFAULT);
    printf("    -n, --num <count>   Number of frames to record (default: %d)\n", RECORD_FRAME_NUMS_DEFAULT);
    printf("    -m, --moment        Enable moment transfer to exoskeleton\n");
    printf("    -h, --help          Show this help message and exit\n\n");
    printf("Examples:\n");
    printf("    # Run inference on left leg with moment transfer\n");
    printf("    %s -i -m -s l model.rknn /dev/ttyUSB0\n\n", prog_name);
    printf("    # Record 5000 frames to custom path\n");
    printf("    %s -r -p /tmp/data.bin -n 5000 model.rknn /dev/ttyUSB0\n\n", prog_name);
    printf("    # Inference and record simultaneously\n");
    printf("    %s -i -r model.rknn /dev/ttyUSB0\n\n", prog_name);
}

int main(int argc, char** argv) {
    // 命令行选项定义
    static const struct option long_options[] = {
        {"side",    required_argument, nullptr, 's'},
        {"infer",   no_argument,       nullptr, 'i'},
        {"record",  no_argument,       nullptr, 'r'},
        {"path",    required_argument, nullptr, 'p'},
        {"num",     required_argument, nullptr, 'n'},
        {"moment",  no_argument,       nullptr, 'm'},
        {"help",    no_argument,       nullptr, 'h'},
        {0,         0,                 0,        0 }
    };

    // 解析选项
    char side = 'l';
    bool do_infer = false;
    bool record_data = false;
    bool transfer_moment = false;
    const char* save_path = nullptr;
    int record_num = -1;
    int opt;

    while ((opt = getopt_long(argc, argv, "s:irp:n:mh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 's':
                if (optarg[0] != 'l' && optarg[0] != 'r') {
                    printf("leg side is assiged wrong! get %s, but require 'r' or 'l'\n", optarg);
                    return 1;
                }
                side = optarg[0];
                printf("Processing for %s leg\n", side == 'l' ? "LEFT" : "RIGHT");
                break;

            case 'r':
                printf("record funtion is on\n");
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
                printf("send moment function is on\n");
                transfer_moment = true;
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
        printf("Error: Missing required arguments (model_path imu_port)!\n");
        printf("Use '%s --help' for usage information.\n", argv[0]);
        return 1;
    }

    if (!save_path) save_path = RECORD_FILE_PATH_DEFAULT;
    if (record_num <= 0) record_num = RECORD_FRAME_NUMS_DEFAULT;

    const char* model_path = argv[optind];
    const std::string imu_port = argv[optind + 1];

    printf("\n=== Configuration ===\n");
    printf("Model path: %s\n", model_path);
    printf("IMU port: %s\n", imu_port.c_str());
    printf("Leg side: %s\n", side == 'l' ? "LEFT" : "RIGHT");
    printf("Inference: %s\n", do_infer ? "enabled" : "disabled");
    printf("Recording: %s\n", record_data ? "enabled" : "disabled");
    if (record_data) {
        printf("\tSave path: %s\n", save_path);
        printf("\tMax frames: %d\n", record_num);
    }
    printf("Moment transfer: %s\n", transfer_moment ? "enabled" : "disabled");
    printf("=====================\n\n");
    
    set_imu_side(side);

    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 启动IMU
    if (!start_imu_driver(imu_port)) {
        printf("start imu failed\n");
        return -1;
    }

    rknn_context ctx;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int ret = model_info_parse(ctx, const_cast<char*>(model_path), input_attrs, output_attrs);

    if (ret != RKNN_SUCC) {
        printf("model info parse failed!\n");
        stop_imu_driver();
        return -1;
    }
    
    if (input_attrs[0].type != RKNN_TENSOR_INT8) {
        printf("Model Input data type should be int8!\n");
        stop_imu_driver();
        return -1;
    }

    rknn_tensor_mem *input_mems[1];
    input_attrs[0].pass_through = 1;
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

    int batch = input_attrs[0].dims[0];
    int channel = 8;
    int h = input_attrs[0].dims[2];
    int w = input_attrs[0].dims[3];
    int expected_frames = h * w;

    // 等待数据流填满
    while (g_running) {
        int cur = 0;
        {
            std::lock_guard<std::mutex> lock(get_data_stream_mutex());
            cur = (int)get_data_stream().size();
        }
        printf("waiting for data stream full [%d/%d]\n", cur, expected_frames);
        if (cur >= expected_frames) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (!g_running) {
        printf("interrupted before data ready\n");
        stop_imu_driver();
        rknn_destroy_mem(ctx, input_mems[0]);
        rknn_destroy(ctx);
        return 0;
    }

    printf("data stream is full, ready to infer\n");

    rknn_tensor_mem *output_mems[1];
    output_mems[0] = nullptr;
    
    if (do_infer) {
        output_mems[0] = rknn_create_mem(ctx, output_attrs[0].size_with_stride);

        ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
        if (ret != RKNN_SUCC) {
            printf("rknn_set_io_mem input fail! ret=%d\n", ret);
            resource_destroy(ctx, input_mems[0], output_mems[0]);
            stop_imu_driver();
            return -1;
        }

        ret = rknn_set_io_mem(ctx, output_mems[0], &output_attrs[0]);
        if (ret != RKNN_SUCC) {
            printf("rknn_set_io_mem output fail! ret=%d\n", ret);
            resource_destroy(ctx, input_mems[0], output_mems[0]);
            stop_imu_driver();
            return -1;
        }
    }

    std::vector<float> src_buffer(batch * channel * expected_frames, 0.0f);
    
    std::ofstream record_file;
    int recorded_count = 0;
    if (record_data) {
        record_file.open(save_path, std::ios::binary);
        if (!record_file.is_open()) {
            printf("Failed to open record file: %s\n", save_path);
            resource_destroy(ctx, input_mems[0], output_mems[0]);
            stop_imu_driver();
            return -1;
        }
        printf("Recording to: %s\n", save_path);
        printf("Data shape: batch=%d, channel=%d, length=%d\n", batch, channel, expected_frames);
    }
    
    if (do_infer && !g_output_filter_initialized) {
        g_output_filter.setup(2, 200.0, 5.0);
        g_output_filter_initialized = true;
    }
    
    // 推理统计
    int infer_count = 0;
    double total_infer_ms = 0.0;
    int moment_send_count = 0;  // 控制 moment 下发频率为 100Hz（每2次推理下发一次）
    
    printf("Starting %s loop (every %dms), press Ctrl+C to stop...\n", 
           do_infer ? (record_data ? "inference+recording" : "inference") : "recording", 
           INFER_INTERVAL_MS);

    while (g_running) {
        auto loop_start = std::chrono::steady_clock::now();
        
        // 1. 从数据流获取最新数据
        {
            std::lock_guard<std::mutex> lock(get_data_stream_mutex());
            const DataStream& ds = get_data_stream();
            
            if ((int)ds.size() < expected_frames) {
                std::this_thread::sleep_for(std::chrono::milliseconds(INFER_INTERVAL_MS));
                continue;
            }
            int offset = ds.size() - expected_frames;
            for (int i = 0; i < expected_frames; ++i) {
                const auto& f = ds[offset + i];
                src_buffer[0 * expected_frames + i] = f.gyro_x;
                src_buffer[1 * expected_frames + i] = f.gyro_y;
                src_buffer[2 * expected_frames + i] = f.gyro_z;
                src_buffer[3 * expected_frames + i] = f.acc_x;
                src_buffer[4 * expected_frames + i] = f.acc_y;
                src_buffer[5 * expected_frames + i] = f.acc_z;
                src_buffer[6 * expected_frames + i] = f.motorPos;
                src_buffer[7 * expected_frames + i] = f.motorVel;
            }
        }

        // only record input data
        if (record_data) {
            record_file.write(reinterpret_cast<const char*>(src_buffer.data()), 
                              src_buffer.size() * sizeof(float));
            recorded_count++;
            
            if (recorded_count % 100 == 0) {
                printf("[Recording] Saved %d frames\n", recorded_count);
            }
            
            if (!do_infer && recorded_count >= record_num) {
                printf("Reached max record frames (%d), stopping...\n", record_num);
                break;
            }
            
            // 控制采样间隔
            if (!do_infer) {
                auto loop_end = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
                if (elapsed < INFER_INTERVAL_MS) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(INFER_INTERVAL_MS - elapsed));
                } else printf("[Warning] fps don't reach to requirement\n");
                continue;
            }
        }

        // 2. 量化并拷贝到输入
        NCHW_float32_to_NC1HWC2_int8(
            src_buffer.data(),
            (int8_t*)input_mems[0]->virt_addr,
            batch, channel, h, w,
            input_attrs[0].scale,
            input_attrs[0].zp,
            g_mean_vals, g_std_vals, channel
        );

        // 3. 推理
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

        // 4. 获取输出并滤波
        float raw_output = (((int32_t)((int8_t*)output_mems[0]->virt_addr)[279*16]) - output_attrs[0].zp) * output_attrs[0].scale;
        float filtered_output = filter_output(raw_output);
        
        // 5. 发送力矩指令 (每2次推理下发一次，保持100Hz)
        moment_send_count++;
        if (transfer_moment && moment_send_count % 2 == 0) send_moment(filtered_output*10.f, 0.0f);

        // 6. 打印统计信息
        if (!record_data && infer_count % 10 == 0) {
            printf("[%5d] gyro=(%.2f, %.2f, %.2f) | acc=(%.2f, %.2f, %.2f) | motor=(%.2f, %.2f)\n",
                   infer_count,
                   src_buffer[1 * expected_frames - 1],  // gyro_x
                   src_buffer[2 * expected_frames - 1],  // gyro_y
                   src_buffer[3 * expected_frames - 1],  // gyro_z
                   src_buffer[4 * expected_frames - 1],  // acc_x
                   src_buffer[5 * expected_frames - 1],  // acc_y
                   src_buffer[6 * expected_frames - 1],  // acc_z
                   src_buffer[7 * expected_frames - 1],  // motorPos
                   src_buffer[8 * expected_frames - 1]); // motorVel
        }

        if (!record_data && infer_count % 20 == 0) {
            double avg_infer_ms = total_infer_ms / infer_count;
            printf("        => raw=%.4f | filtered=%.4f Nm/kg | avg=%.2fms | %.1f fps\n",
                   raw_output, filtered_output, avg_infer_ms, 1000.0 / avg_infer_ms);
        }

        // 如果同时记录，保存滤波后的输出
        if (record_data) {
            record_file.write(reinterpret_cast<const char*>(&filtered_output), sizeof(float));
            if (recorded_count >= record_num) {
                printf("Reached max record frames (%d), stopping...\n", record_num);
                break;
            }
        }

        // 7. 控制推理间隔为10ms
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
        if (elapsed < INFER_INTERVAL_MS) std::this_thread::sleep_for(std::chrono::milliseconds(INFER_INTERVAL_MS - elapsed));
        else printf("[Warning] fps don't reach to requirement\n");
    }

    // 关闭记录文件
    if (record_data && record_file.is_open()) {
        record_file.close();
        printf("\n=== Recording Complete ===\n");
        printf("Total frames recorded: %d\n", recorded_count);
        printf("File saved to: %s\n", save_path);
        printf("Data size: %.2f MB\n", 
               do_infer ? ((recorded_count * (channel*expected_frames+1)) * sizeof(float)) / (1024.0 * 1024.0) :
               (recorded_count * channel * expected_frames * sizeof(float)) / (1024.0 * 1024.0));
    }

    // 打印最终统计
    if (do_infer && infer_count > 0) {
        printf("\n=== Final Statistics ===\n");
        printf("Total inferences: %d\n", infer_count);
        printf("Average inference time: %.3f ms\n", total_infer_ms / infer_count);
        printf("Average FPS: %.1f\n", 1000.0 / (total_infer_ms / infer_count));
    }

    send_moment(0.0f, 0.0f);
    resource_destroy(ctx, input_mems[0], output_mems[0]);
    stop_imu_driver();
    
    printf("Program exited cleanly.\n");
    return 0;
}