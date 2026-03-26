#include "rknn_api.h"
#include "model_process/model_process.h"
#include "data_process/hip_data_producer.h"
#include "timing/periodic_timer.h"
#include "fp16/Float16.h"
#include <boost/circular_buffer.hpp>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <string>
#include <thread>
#include "DspFilters/Butterworth.h"

static Dsp::SimpleFilter<Dsp::Butterworth::LowPass<2>, 1> g_output_filter;
static bool g_output_filter_initialized = false;

static float filter_output(float value) {
    float* channels[1] = {&value};
    g_output_filter.process(1, channels);
    return value;
}
// 推理间隔 (ms)--200Hz
constexpr int INFER_INTERVAL_MS = 5;
constexpr int FILL_CHECK_INTERVAL_MS = 50;

static std::atomic<bool> g_running{true};

static void signal_handler(int signum)
{
    printf("\nReceived signal %d, stopping...\n", signum);
    g_running = false;
}

void print_usage(const char *prog_name)
{
    printf("Usage: %s [options] <model_path> <serial_port>\n\n", prog_name);
    printf("Description:\n");
    printf("    Run hip torque regression model inference.\n\n");
    printf("Positional Arguments:\n");
    printf("    model_path          Path to the RKNN model file\n");
    printf("    serial_port         Serial port for MCU communication (e.g., /dev/ttyS3)\n\n");
    printf("Options:\n");
    printf("    -s, --side <l|r>    Leg side: 'l' for left, 'r' for right (default: l)\n");
    printf("    -m, --moment        Enable torque transfer to exoskeleton\n");
    printf("    -d, --debug         Debug to show more information\n");
    printf("    -h, --help          Show this help message and exit\n\n");
    printf("Examples:\n");
    printf("    # Run torque inference on left leg with torque transfer\n");
    printf("    %s -m -s l model.rknn /dev/ttyS3\n\n", prog_name);
}

int main(int argc, char **argv)
{
    static const struct option long_options[] = {
        {"side", required_argument, nullptr, 's'},
        {"moment", no_argument, nullptr, 'm'},
        {"debug", no_argument, nullptr, 'd'},
        {"help", no_argument, nullptr, 'h'},
        {0, 0, 0, 0}};

    char side = 'l';
    bool transfer_moment = false;
    bool debug = false;
    int opt;

    while ((opt = getopt_long(argc, argv, "s:mdh", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 's':
            if (optarg[0] != 'l' && optarg[0] != 'r')
            {
                printf("leg side is assigned wrong! get %s, but require 'r' or 'l'\n", optarg);
                return 1;
            }
            side = optarg[0];
            printf("Processing for %s leg\n", side == 'l' ? "LEFT" : "RIGHT");
            break;

        case 'm':
            printf("moment transfer function is on\n");
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

    if (optind + 2 > argc)
    {
        printf("Error: Missing required arguments (model_path serial_port)!\n");
        printf("Use '%s --help' for usage information.\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[optind];
    const std::string serial_port = argv[optind + 1];

    printf("\n=== Configuration ===\n");
    printf("Model path: %s\n", model_path);
    printf("Serial port: %s\n", serial_port.c_str());
    printf("Leg side: %s\n", side == 'l' ? "LEFT" : "RIGHT");
    printf("Inference: %s\n", "enabled");
    printf("Moment transfer: %s\n", transfer_moment ? "enabled" : "disabled");
    printf("Debug mode: %s\n", debug ? "enabled" : "disabled");
    printf("=====================\n\n");

    // hip_producer::set_leg_side(side);

    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 启动串口通信
    if (!hip_producer::start_serial2mcu(serial_port))
    {
        printf("start serial2mcu failed\n");
        return -1;
    }

    rknn_context ctx;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int ret = model_info_parse(ctx, const_cast<char *>(model_path), input_attrs, output_attrs);

    if (ret != RKNN_SUCC)
    {
        printf("model info parse failed!\n");
        hip_producer::stop_serial2mcu();
        return -1;
    }

    input_attrs[0].pass_through = 1;

    assert(Hip::checkModelInfo(input_attrs));
    assert(hip_producer::data_stream.is_lock_free());

    std::vector<rknn_tensor_mem *> input_mems;
    std::vector<rknn_tensor_mem *> output_mems;
    ret = alllocate_set_io_memory(ctx, input_attrs, output_attrs, input_mems, output_mems);
    if (ret != RKNN_SUCC)
    {
        printf("alllocate_set_io_memory fail! ret=%d\n", ret);
        resource_destroy(ctx, input_mems, output_mems);
        hip_producer::stop_serial2mcu();
        return -1;
    }

    int infer_count = 0;
    double total_infer_ms = 0.0;
    int moment_send_count = 0;
    if (!g_output_filter_initialized) {
        g_output_filter.setup(2, 200.0, 5.0);
        g_output_filter_initialized = true;
    }

    timing::PeriodicTimer fill_timer;
    if (!fill_timer.start(FILL_CHECK_INTERVAL_MS))
    {
        printf("create fill timerfd failed\n");
        hip_producer::stop_serial2mcu();
        resource_destroy(ctx, input_mems, output_mems);
        return -1;
    }

    boost::circular_buffer<Hip::DataFrame> src_buffer(Hip::STREAM_LENGTH);
    while (g_running)
    {
        int cur = (int)hip_producer::data_stream.read_available();
        if (cur > 0) {
            hip_producer::data_stream.consume_all([&src_buffer](Hip::DataFrame frame)
                              { src_buffer.push_back(frame); });
        }
        printf("waiting for data stream full [%d/%d]\n", src_buffer.size(), Hip::STREAM_LENGTH);
        if (src_buffer.full())
            break;
        uint64_t missed = 0;
        if (!fill_timer.wait(missed, &g_running))
        {
            printf("fill timer wait failed\n");
            hip_producer::stop_serial2mcu();
            resource_destroy(ctx, input_mems, output_mems);
            return -1;
        }
    }
    fill_timer.stop();

    if (!g_running)
    {
        printf("interrupted before data ready\n");
        hip_producer::stop_serial2mcu();
        rknn_destroy(ctx);
        return 0;
    }

    printf("data stream is full, ready to infer\n");

    printf("Starting inference loop (every %dms), press Ctrl+C to stop...\n", INFER_INTERVAL_MS);
    timing::PeriodicTimer infer_timer;
    if (!infer_timer.start(INFER_INTERVAL_MS, 1))
    {
        printf("create infer timerfd failed\n");
        hip_producer::stop_serial2mcu();
        resource_destroy(ctx, input_mems, output_mems);
        return -1;
    }
    
    while (g_running)
    {
        uint64_t missed = 0;
        if (!infer_timer.wait(missed, &g_running))
        {
            printf("infer timer wait failed\n");
            break;
        }
        if (missed > 1)
        {
            printf("[Warning] Overrun! Missed %llu pulses\n", (unsigned long long)(missed - 1));
        }

        if (hip_producer::data_stream.read_available())
        {
            hip_producer::data_stream.consume_all([&src_buffer](Hip::DataFrame frame)
                                      { src_buffer.push_back(frame); });
        }

        ret = fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8<Hip::DataFrame>(src_buffer,
                                                                (int8_t *)input_mems[0]->virt_addr,
                                                                Hip::STREAM_LENGTH,
                                                                input_attrs[0].scale,
                                                                input_attrs[0].zp,
                                                                Hip::g_mean_vals,
                                                                Hip::g_std_vals,
                                                                Hip::ModelChannel);
        if (ret < 0)
        {
            printf("data buffer is not full\n");
            continue;
        }

        auto infer_start = std::chrono::steady_clock::now();
        ret = rknn_run(ctx, NULL);
        auto infer_end = std::chrono::steady_clock::now();

        if (ret != RKNN_SUCC)
        {
            printf("rknn run error %d\n", ret);
            break;
        }

        double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        total_infer_ms += infer_ms;
        ++infer_count;

        float raw_output = (((int32_t)((int8_t *)output_mems[0]->virt_addr)[Hip::STREAM_LENGTH - 1]) - output_attrs[0].zp) * output_attrs[0].scale;
        float filtered_output = filter_output(raw_output);

        ++moment_send_count;
        if (transfer_moment && moment_send_count % 2 == 0)
            hip_producer::sendData(filtered_output, 0x00, 0x00);

        if (debug && infer_count % 50 == 0)
        {
            double avg_infer_ms = total_infer_ms / infer_count;
            printf("[%5d] raw=%.4f | torque=%.4f | avg=%.2fms | %.1f fps\n",
                   infer_count, raw_output, filtered_output,
                   avg_infer_ms, 1000.0 / avg_infer_ms);
        }
    }
    infer_timer.stop();

    // 打印最终统计
    if (infer_count > 0)
    {
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
