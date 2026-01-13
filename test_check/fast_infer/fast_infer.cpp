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
constexpr int INFER_INTERVAL_MS = 5;   // 推理间隔 (ms)--200Hz


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
    printf("    Run TCN model inference on IMU data.\n\n");
    printf("Positional Arguments:\n");
    printf("    model_path          Path to the RKNN model file\n");
    printf("    imu_port            Serial port for IMU device (e.g., /dev/ttyUSB0)\n\n");
    printf("Options:\n");
    printf("    -s, --side <l|r>    Leg side: 'l' for left, 'r' for right (default: l)\n");
    printf("    -m, --moment        Enable moment transfer to exoskeleton\n");
    printf("    -d, --debug         Debug to show more information\n");
    printf("    -h, --help          Show this help message and exit\n\n");
    printf("Examples:\n");
    printf("    # Run inference on left leg with moment transfer\n");
    printf("    %s -m -s l model.rknn /dev/ttyUSB0\n\n", prog_name);
}

int main(int argc, char** argv) {
    // 命令行选项定义
    static const struct option long_options[] = {
        {"side",    required_argument, nullptr, 's'},
        {"moment",  no_argument,       nullptr, 'm'},
        {"debug",   no_argument,       nullptr, 'd'},
        {"help",    no_argument,       nullptr, 'h'},
        {0,         0,                 0,        0 }
    };

    // 解析选项
    char side = 'l';
    bool transfer_moment = false;
    bool debug = false;
    int opt;

    while ((opt = getopt_long(argc, argv, "s:mdh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 's':
                if (optarg[0] != 'l' && optarg[0] != 'r') {
                    printf("leg side is assiged wrong! get %s, but require 'r' or 'l'\n", optarg);
                    return 1;
                }
                side = optarg[0];
                printf("Processing for %s leg\n", side == 'l' ? "LEFT" : "RIGHT");
                break;
            
            case 'm':
                printf("send moment function is on\n");
                transfer_moment = true;
                break;

            case 'd':
                printf("debug mode is on\n");
                debug = true;
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

    if (optind + 2 > argc) {
        printf("Error: Missing required arguments (model_path imu_port)!\n");
        printf("Use '%s --help' for usage information.\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[optind];
    const std::string imu_port = argv[optind + 1];

    printf("\n=== Configuration ===\n");
    printf("Model path: %s\n", model_path);
    printf("IMU port: %s\n", imu_port.c_str());
    printf("Leg side: %s\n", side == 'l' ? "LEFT" : "RIGHT");
    printf("Inference: %s\n", "enabled");
    printf("Moment transfer: %s\n", transfer_moment ? "enabled" : "disabled");
    printf("Debug mode: %s\n", debug ? "enabled" : "disabled");
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
    input_attrs[0].pass_through = 1;

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
        rknn_destroy(ctx);
        return 0;
    }

    printf("data stream is full, ready to infer\n");

    rknn_tensor_mem *input_mems[1];
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

    rknn_tensor_mem *output_mems[1];
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
    
    if (!g_output_filter_initialized) {
        g_output_filter.setup(2, 200.0, 5.0);
        g_output_filter_initialized = true;
    }
    
    // 推理统计
    int infer_count = 0;
    double total_infer_ms = 0.0;
    int moment_send_count = 0;  // 控制moment下发频率为100Hz（每2次推理下发一次）
    
    printf("Starting inference loop (every %dms), press Ctrl+C to stop...\n", INFER_INTERVAL_MS);

    while (g_running) {
        auto loop_start = std::chrono::steady_clock::now();
        
        // 从数据流获取最新数据
        get_data_stream_mutex().lock();
        const DataStream& ds = get_data_stream();
        
        if ((int)ds.size() < expected_frames) {
            get_data_stream_mutex().unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(INFER_INTERVAL_MS));
            continue;
        }
        fast_NHWC_float32_deque_to_NC1HWC2_int8(ds, (int8_t*)input_mems[0]->virt_addr, w, 
                        input_attrs[0].scale, input_attrs[0].zp, g_mean_vals, g_std_vals, 8);
        get_data_stream_mutex().unlock();

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

        // 获取输出并滤波
        float raw_output = (((int32_t)((int8_t*)output_mems[0]->virt_addr)[279*16]) - output_attrs[0].zp) * output_attrs[0].scale;
        float filtered_output = filter_output(raw_output);
        
        // 发送力矩指令 (每2次推理下发一次，保持100Hz)
        moment_send_count++;
        if (transfer_moment && moment_send_count % 2 == 0) send_moment(filtered_output*14.f, 0.0f);


        if (debug && infer_count % 50 == 0) {
            double avg_infer_ms = total_infer_ms / infer_count;
            printf("        => raw=%.4f | filtered=%.4f Nm/kg | avg=%.2fms | %.1f fps\n",
                   raw_output, filtered_output, avg_infer_ms, 1000.0 / avg_infer_ms);
        }

        // 控制推理间隔
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
        if (elapsed < INFER_INTERVAL_MS) std::this_thread::sleep_for(std::chrono::milliseconds(INFER_INTERVAL_MS - elapsed));
        else printf("[Warning] fps don't reach to requirement\n");
    }


    // 打印最终统计
    if (infer_count > 0) {
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