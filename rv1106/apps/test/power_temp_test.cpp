#include "rknn_api.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "model_process/model_info_parse.hpp"
#include "fp16/Float16.h"
#include <csignal>
#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include "serial/uart_serial.h"
#include <getopt.h>
#include <atomic>

constexpr uint8_t DATA_HEAD1 = 0x7C;
constexpr uint8_t DATA_HEAD2 = 0xF4;
constexpr uint8_t DATA_END1  = 0xE3;
constexpr uint8_t DATA_END2  = 0xEC;
static std::unique_ptr<SerialLoop<20>> g_driver;

float readChipTemperature() {
    const std::string tempFilePath = "/sys/class/thermal/thermal_zone0/temp";
    std::ifstream file(tempFilePath);

    if (!file.is_open()) {
        std::cerr << "错误：无法打开温度文件 " << tempFilePath << std::endl;
        return -273.15;
    }

    int millideg;
    file >> millideg;

    if (file.fail()) {
        std::cerr << "错误：从温度文件读取数据失败" << std::endl;
        return -273.15;
    }

    file.close();
    return millideg / 1000.0; // 转换为摄氏度
}

bool start_driver(const std::string& portName) {
    std::vector<uint8_t> head = {DATA_HEAD1, DATA_HEAD2};
    std::vector<uint8_t> end = {DATA_END1, DATA_END2};
    timeval timeout = {0, 15000};  // 15ms超时
    
    g_driver = SerialLoop<20>::create(portName, head, end, B115200, timeout);
    if (!g_driver) {
        return false;
    }

    g_driver->setCallback([](const uint8_t* frame) {
        static int count = 0;
        float mcu_temp;
        memcpy(&mcu_temp, frame, 4);
        if (count++ % 300 == 0) {
            float soc_temp = readChipTemperature();
            printf("mcu temp is %.3f, soc temp is %.3f\n", mcu_temp, soc_temp);
        }
    });

    g_driver->start();
    return true;
}

void stop_driver() {
    if (g_driver) {
        g_driver->stop();
        g_driver.reset();
    }
}


volatile bool g_exit_flag = false;
static void stop(int signum)
{
    printf("\nReceived signal %d, stopping...\n", signum);
    g_exit_flag = true;
}
static void resource_destroy(rknn_context ctx, rknn_tensor_mem *input_mem, rknn_tensor_mem *output_mem)
{
    if (input_mem)
        rknn_destroy_mem(ctx, input_mem);
    if (output_mem)
        rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
}

static const float g_mean_vals[8] = {
    0.0f, 0.0f, 0.0f, // gyro
    0.0f, 0.0f, 0.0f, // acc
    0.0f, 0.0f        // motor
};
static const float g_std_vals[8] = {
    1.0f, 1.0f, 1.0f, // gyro
    1.0f, 1.0f, 1.0f, // acc
    1.0f, 1.0f        // motor
};


void model_run(rknn_context &ctx, rknn_tensor_mem **input_mems, rknn_tensor_mem **output_mems, float *inputs, std::vector<rknn_tensor_attr> const &input_attrs) {
    unsigned long long infer_count = 0;
    double total_infer_ms = 0.0;
    while (!g_exit_flag) {
        
        NCHW_float32_to_NC1HWC2_int8(inputs, (int8_t *)input_mems[0]->virt_addr, 1, 8,
                                    input_attrs[0].dims[2], input_attrs[0].dims[3], input_attrs[0].scale, input_attrs[0].zp, g_mean_vals, g_std_vals, 8);

        auto loop_start = std::chrono::steady_clock::now();
        rknn_run(ctx, NULL);
        auto infer_end = std::chrono::steady_clock::now();
        
        ++infer_count;
        double infer_ms = std::chrono::duration<double, std::milli>(infer_end - loop_start).count();
        total_infer_ms += infer_ms;
     
        
        auto loop_end = std::chrono::steady_clock::now();
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
        if (elapsed < 10) std::this_thread::sleep_for(std::chrono::milliseconds(10 - elapsed));
        else printf("[Warning] fps don't reach to requirement\n");
    }
    if (infer_count > 0) {
        printf("\n=== Final Statistics ===\n");
        printf("Total inferences: %llu\n", infer_count);
        printf("Average inference time: %.3f ms\n", total_infer_ms / infer_count);
        printf("Average FPS: %.1f\n", 1000.0 / (total_infer_ms / infer_count));
    }
}

void flash_run(int const cycle, int const capacity) {
    const std::string flash_file = "/root/cycle_test";
    // 确保文件存在（如不存在则创建），再以读写二进制模式打开
    {
        std::ofstream init(flash_file, std::ios::binary | std::ios::app);
        if (!init.is_open()) {
            fprintf(stderr, "[Flash] Failed to create %s\n", flash_file.c_str());
            return;
        }
    }
    std::fstream record_file(flash_file, std::ios::in | std::ios::out | std::ios::binary);
    if (!record_file.is_open()) {
        fprintf(stderr, "[Flash] Failed to open %s\n", flash_file.c_str());
        return;
    }

    const int16_t save_data = static_cast<int16_t>(0xAAAAu);
    const int target_items = capacity / static_cast<int>(sizeof(save_data));

    if (cycle <= 0 || target_items <= 0) {
        printf("[Flash] Nothing to do: cycle=%d, capacity=%d\n", cycle, capacity);
        g_exit_flag = true;
        record_file.close();
        return;
    }

    for (int i = 0; i < cycle && !g_exit_flag; ++i) {
        // --- 写入 ---
        record_file.clear();
        record_file.seekp(0, std::ios::beg);
        int written_items = 0;
        for (int j = 0; j < target_items && !g_exit_flag; ++j) {
            record_file.write(reinterpret_cast<const char *>(&save_data), sizeof(save_data));
            if (!record_file) {
                fprintf(stderr, "[Flash] Write failed at pos %d\n", j);
                break;
            }
            ++written_items;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        record_file.flush();

        if (written_items == 0) {
            printf("[Flash] Cycle %d: no data written, skip verify\n", i);
        } else {
            // --- 校验写入数据（中断时只校验已写入位置）---
            record_file.clear();
            record_file.seekg(0, std::ios::beg);
            bool data_ok = true;
            int16_t read_data = 0;
            for (int j = 0; j < written_items; ++j) {
                record_file.read(reinterpret_cast<char *>(&read_data), sizeof(read_data));
                if (!record_file) {
                    data_ok = false;
                    printf("[Flash] Cycle %d: READ FAILED at pos %d (verified %d/%d items)\n",
                           i, j, j, written_items);
                    break;
                }
                if (read_data != save_data) {
                    data_ok = false;
                    printf("[Flash] Cycle %d: DATA MISMATCH at pos %d, expected 0x%04X got 0x%04X\n",
                           i, j,
                           static_cast<uint16_t>(save_data),
                           static_cast<uint16_t>(read_data));
                    break;
                }
            }

            if (data_ok) {
                if (written_items == target_items)
                    printf("[Flash] Cycle %d: OK (%d items)\n", i, written_items);
                else
                    printf("[Flash] Cycle %d: INTERRUPTED, verified %d/%d items\n",
                           i, written_items, target_items);
            }
        }

        // --- 清理：以 trunc 模式重新打开，将文件截断为 0 字节 ---
        record_file.close();
        record_file.open(flash_file,
                         std::ios::in | std::ios::out |
                         std::ios::binary | std::ios::trunc);
        if (!record_file.is_open()) {
            fprintf(stderr, "[Flash] Failed to reopen file for truncation\n");
            g_exit_flag = true;
            return;
        }
    }

    record_file.close();
    if (!g_exit_flag)
        printf("[Flash] Reached cycle limit (%d), stopping all threads.\n", cycle);
    g_exit_flag = true;
}


int main(int argc, char **argv)
{
    static const struct option long_options[] = {
        {"help", no_argument, nullptr, 'h'},
        {0, 0, 0, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "h", long_options, nullptr)) != -1) {
        switch (opt)
        {
        case 'h':
            printf("Usage: %s <model_path> [cycle] [capacity] [-h/--help]\n", argv[0]);
            return 0;
        case '?':
        default:
            printf("Error: Unknown option\n");
            printf("Usage: %s <model_path> [cycle] [capacity] [-h/--help]\n", argv[0]);
            return 1;
        }
    }
    if (optind >= argc) {
        printf("Error: model path is required\n");
        printf("Usage: %s <model_path> [cycle] [capacity] [-h/--help]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[optind];
    int cycle = 1000;
    int capacity = 600000;
    if (optind + 2 < argc) {
        cycle    = atoi(argv[optind + 1]);
        capacity = atoi(argv[optind + 2]);
    } else if (optind + 1 < argc) {
        cycle = atoi(argv[optind + 1]);
    }
    printf("Model: %s, Cycle: %d, Capacity: %d bytes\n", model_path, cycle, capacity);
    start_driver("/dev/ttyS0");
    rknn_context ctx;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int ret = model_info_parse(ctx, const_cast<char *>(model_path), input_attrs, output_attrs);

    if (ret != RKNN_SUCC)
    {
        printf("model info parse failed!\n");
        return -1;
    }

    input_attrs[0].pass_through = 1;
    int h = input_attrs[0].dims[2];
    int w = input_attrs[0].dims[3];
    int expected_frames = h * w;

    rknn_tensor_mem *input_mems[1];
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    rknn_tensor_mem *output_mems[1];
    output_mems[0] = rknn_create_mem(ctx, output_attrs[0].size_with_stride);

    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    if (ret != RKNN_SUCC)
    {
        printf("rknn_set_io_mem input fail! ret=%d\n", ret);
        resource_destroy(ctx, input_mems[0], output_mems[0]);
        return -1;
    }

    ret = rknn_set_io_mem(ctx, output_mems[0], &output_attrs[0]);
    if (ret != RKNN_SUCC)
    {
        printf("rknn_set_io_mem output fail! ret=%d\n", ret);
        resource_destroy(ctx, input_mems[0], output_mems[0]);
        return -1;
    }
    float *inputs = new float[expected_frames * 8];
    
    signal(SIGINT, stop);
    signal(SIGTERM, stop);

    std::thread model_run_thread, flash_run_thread;

    model_run_thread = std::thread(model_run, std::ref(ctx), input_mems, output_mems, inputs, input_attrs);
    flash_run_thread = std::thread(flash_run, cycle, capacity);

    if (flash_run_thread.joinable()) flash_run_thread.join();
    if (model_run_thread.joinable()) model_run_thread.join();
    stop_driver();
    resource_destroy(ctx, input_mems[0], output_mems[0]);
    delete[] inputs;
    return 0;
}