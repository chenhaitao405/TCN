#include "rknn_api.h"
#include "model_process/model_process.h"
#include "data_process/knee_data_producer.h"
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
#include "timing/periodic_timer.h"
#include "non_linear_filter/non_linear_filter.hpp"

static Dsp::SimpleFilter<Dsp::Butterworth::LowPass<2>, 1> g_output_filter;
static bool g_output_filter_initialized = false;

static float filter_output(float value) {
    float* channels[1] = {&value};
    g_output_filter.process(1, channels);
    return value;
}

constexpr int RECORD_FRAME_NUMS_DEFAULT = 5000;
constexpr int INFER_INTERVAL_MS = 5;
constexpr int FILL_CHECK_INTERVAL_MS = 50;
constexpr const char* RECORD_FILE_PATH_DEFAULT = "/data/knee_record_data.bin";

static std::atomic<bool> g_running{true};

static void signal_handler(int signum) {
    g_running.store(false, std::memory_order_release);
}

void print_usage(const char* prog_name) {
    printf("Usage: %s [options] <model_path> <imu_port>\n\n", prog_name);
    printf("Description:\n");
    printf("    Run knee torque regression model inference or record raw sensor data.\n\n");
    printf("Positional Arguments:\n");
    printf("    model_path          Path to the RKNN model file\n");
    printf("    imu_port            Serial port for IMU device (e.g., /dev/ttyUSB0)\n\n");
    printf("Options:\n");
    printf("    -s, --side <l|r>    Leg side: 'l' for left, 'r' for right (default: l)\n");
    printf("    -i, --infer         Enable inference mode\n");
    printf("    -r, --record        Enable recording mode (save raw IMU data)\n");
    printf("    -p, --path <path>   Path to save recorded data (default: %s)\n", RECORD_FILE_PATH_DEFAULT);
    printf("    -n, --num <count>   Number of frames to record (default: %d)\n", RECORD_FRAME_NUMS_DEFAULT);
    printf("    -m, --moment        Enable torque transfer to exoskeleton\n");
    printf("    -d, --debug         Debug to show more information\n");
    printf("    -h, --help          Show this help message and exit\n\n");
    printf("Examples:\n");
    printf("    %s -i -m -s l model.rknn /dev/ttyUSB0\n\n", prog_name);
    printf("    %s -r -p /tmp/data.bin -n 5000 model.rknn /dev/ttyUSB0\n\n", prog_name);
    printf("    %s -i -r model.rknn /dev/ttyUSB0\n\n", prog_name);
}

int main(int argc, char** argv) {
    static const struct option long_options[] = {
        {"side",    required_argument, nullptr, 's'},
        {"infer",   no_argument,       nullptr, 'i'},
        {"record",  no_argument,       nullptr, 'r'},
        {"path",    required_argument, nullptr, 'p'},
        {"num",     required_argument, nullptr, 'n'},
        {"moment",  no_argument,       nullptr, 'm'},
        {"debug",   no_argument,       nullptr, 'd'},
        {"help",    no_argument,       nullptr, 'h'},
        {0,          0,                 0,        0 }
    };

    char side = 'l';
    bool do_infer = false;
    bool record_data = false;
    bool debug = false;
    bool transfer_moment = false;
    const char* save_path = nullptr;
    int record_num = -1;
    int opt;

    while ((opt = getopt_long(argc, argv, "s:irp:n:mdh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 's':
                if (optarg[0] != 'l' && optarg[0] != 'r') {
                    printf("leg side is assigned wrong! get %s, but require 'r' or 'l'\n", optarg);
                    return 1;
                }
                side = optarg[0];
                printf("Processing for %s leg\n", side == 'l' ? "LEFT" : "RIGHT");
                break;

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
    printf("Debug mode: %s\n", debug ? "enabled" : "disabled");
    printf("=====================\n\n");

    knee_producer::set_leg_side(side);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (!knee_producer::start_serial2mcu(imu_port)) {
        printf("start serial2mcu failed\n");
        return -1;
    }

    rknn_context ctx;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int ret = model_info_parse(ctx, const_cast<char*>(model_path), input_attrs, output_attrs);

    if (ret != RKNN_SUCC) {
        printf("model info parse failed!\n");
        knee_producer::stop_serial2mcu();
        return -1;
    }

    input_attrs[0].pass_through = 1;

    assert(Knee::checkModelInfo(input_attrs));
    assert(knee_producer::data_stream.is_lock_free());

    std::vector<rknn_tensor_mem *> input_mems;
    std::vector<rknn_tensor_mem *> output_mems;

    if (do_infer) {
        ret = alllocate_set_io_memory(ctx, input_attrs, output_attrs, input_mems, output_mems);
        if (ret != RKNN_SUCC) {
            printf("alllocate_set_io_memory fail! ret=%d\n", ret);
            resource_destroy(ctx, input_mems, output_mems);
            knee_producer::stop_serial2mcu();
            return -1;
        }
    }

    std::ofstream record_file;
    int recorded_count = 0;
    if (record_data) {
        record_file.open(save_path, std::ios::binary);
        if (!record_file.is_open()) {
            printf("Failed to open record file: %s\n", save_path);
            resource_destroy(ctx, input_mems, output_mems);
            knee_producer::stop_serial2mcu();
            return -1;
        }
        printf("Recording to: %s\n", save_path);
        printf("Data shape: channel=%d, length=%d\n", Knee::ModelChannel, Knee::STREAM_LENGTH);
    }

    timing::PeriodicTimer fill_timer;
    if (!fill_timer.start(FILL_CHECK_INTERVAL_MS)) {
		printf("create fill timerfd failed\n");
		knee_producer::stop_serial2mcu();
		resource_destroy(ctx, input_mems, output_mems);
		return -1;
	}

    boost::circular_buffer<Knee::DataFrame> src_buffer(Knee::STREAM_LENGTH);
    while (g_running.load(std::memory_order_acquire)) {
        int cur = (int)knee_producer::data_stream.read_available();
        if (cur > 0) {
            knee_producer::data_stream.consume_all([&src_buffer](Knee::DataFrame frame)
                                                   { src_buffer.push_back(frame); });
        }
        printf("waiting for data stream full [%d/%d]\n", src_buffer.size(), Knee::STREAM_LENGTH);
        if (src_buffer.full())
            break;
        uint64_t missed = 0;
		if (!fill_timer.wait(missed, &g_running)) {
			printf("fill timer wait failed\n");
			knee_producer::stop_serial2mcu();
			resource_destroy(ctx, input_mems, output_mems);
			return -1;
		}
    }
    fill_timer.stop();

    if (!g_running.load(std::memory_order_acquire)) {
        printf("interrupted before data ready\n");
        knee_producer::stop_serial2mcu();
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
    NonLinearFilter & non_linear_filter = NonLinearFilter::getInstance();
    timing::PeriodicTimer infer_timer;
	if (!infer_timer.start(INFER_INTERVAL_MS, 1)) {
		printf("create infer timerfd failed\n");
		knee_producer::stop_serial2mcu();
		resource_destroy(ctx, input_mems, output_mems);
		return -1;
	}

    printf("Starting %s loop (every %dms), press Ctrl+C to stop...\n",
           do_infer ? (record_data ? "inference+recording" : "inference") : "recording",
           INFER_INTERVAL_MS);

    while (g_running.load(std::memory_order_relaxed)) {
        uint64_t missed = 0;
		if (!infer_timer.wait(missed, &g_running)) {
			printf("infer timer wait failed\n");
			break;
		}
		if (missed > 1) {
			printf("[Warning] Overrun! Missed %llu pulses\n", (unsigned long long)(missed - 1));
		}

        if (knee_producer::data_stream.read_available()) {
            knee_producer::data_stream.consume_all([&src_buffer](Knee::DataFrame frame) {
                src_buffer.push_back(frame);
            });
        }

        if (record_data) {
            std::pair<Knee::DataFrame *, std::size_t> a1 = src_buffer.array_one();
            std::pair<Knee::DataFrame *, std::size_t> a2 = src_buffer.array_two();
            if (a1.second > 0)
                record_file.write(reinterpret_cast<const char*>(a1.first), a1.second * sizeof(Knee::DataFrame));
            if (a2.second > 0)
                record_file.write(reinterpret_cast<const char*>(a2.first), a2.second * sizeof(Knee::DataFrame));
            recorded_count++;

            if (recorded_count % 100 == 0) {
                printf("[Recording] Saved %d frames\n", recorded_count);
            }

            if (!do_infer && recorded_count >= record_num) {
                printf("Reached max record frames (%d), stopping...\n", record_num);
                break;
            }

            if (!do_infer) {
                continue;
            }
        }

        ret = fast_NHWC_float32_circular_buffer_to_NC1HWC2_int8<Knee::DataFrame>(
            src_buffer,
            (int8_t*)input_mems[0]->virt_addr,
            Knee::STREAM_LENGTH,
            input_attrs[0].scale,
            input_attrs[0].zp,
            Knee::g_mean_vals,
            Knee::g_std_vals,
            Knee::ModelChannel
        );
        if (ret < 0) continue;
        

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
        float raw_output = (((int32_t)((int8_t*)output_mems[0]->virt_addr)[Knee::STREAM_LENGTH - 1]) - output_attrs[0].zp) * output_attrs[0].scale;
        float filtered_output = filter_output(raw_output);
        if (side == 'r')
			filtered_output *= -1;
        filtered_output = non_linear_filter.nonlinear_filter_output(filtered_output);
		if (src_buffer.back().motorPos > 0.0f) filtered_output = 0.0f;

        // 发送力矩指令 (每2次推理下发一次，保持100Hz)
        moment_send_count++;
        if (transfer_moment && moment_send_count % 2 == 0)
            knee_producer::sendMoment(filtered_output * 12, 0.0f);

        if (debug && infer_count % 20 == 0) {
            const auto& last = src_buffer.back();
            printf("[%5d] gyro=(%.2f, %.2f, %.2f) | acc=(%.2f, %.2f, %.2f) | motor=(%.2f, %.2f)\n",
                   infer_count,
                   last.gyro_x, last.gyro_y, last.gyro_z,
                   last.acc_x, last.acc_y, last.acc_z,
                   last.motorPos, last.motorVel);

            double avg_infer_ms = total_infer_ms / infer_count;
            printf("        => raw=%.4f | torque=%.4f | avg=%.2fms | %.1f fps\n",
                   raw_output, filtered_output, avg_infer_ms, 1000.0 / avg_infer_ms);
        }

        if (record_data) {
            record_file.write(reinterpret_cast<const char*>(&raw_output), sizeof(float));
            record_file.write(reinterpret_cast<const char*>(&filtered_output), sizeof(float));
            if (recorded_count >= record_num) {
                printf("Reached max record frames (%d), stopping...\n", record_num);
                break;
            }
        }
    }

    if (record_data && record_file.is_open()) {
        record_file.close();
        printf("\n=== Recording Complete ===\n");
        printf("Total frames recorded: %d\n", recorded_count);
        printf("File saved to: %s\n", save_path);
        printf("Data size: %.2f MB\n",
               do_infer ? (recorded_count * Knee::STREAM_LENGTH * sizeof(Knee::DataFrame) + recorded_count * (2 * sizeof(float))) / (1024.0 * 1024.0) :
                          (recorded_count * Knee::STREAM_LENGTH * sizeof(Knee::DataFrame)) / (1024.0 * 1024.0));
    }

    if (do_infer && infer_count > 0) {
        printf("\n=== Final Statistics ===\n");
        printf("Total inferences: %d\n", infer_count);
        printf("Average inference time: %.3f ms\n", total_infer_ms / infer_count);
        printf("Average FPS: %.1f\n", 1000.0 / (total_infer_ms / infer_count));
    }

    knee_producer::sendMoment(0.0f, 0.0f);
    knee_producer::stop_serial2mcu();
    resource_destroy(ctx, input_mems, output_mems);

    printf("Program exited cleanly.\n");
    return 0;
}
