#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <poll.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include "gpio_uart/gpio.h"
#include "lsm6dsr.h"

#define I2C_BUS_ID      2
#define LSM_ADDR        0x6A // SDO=0
#define INT1_GPIO_PIN   42   // GPIO1_B2 : (1*32) + (1*8) + 2 = 42

std::atomic<bool> running(true);

// Thread function - Polling mode (no GPIO interrupt)
// Uses adaptive sleep based on FIFO fill rate
void data_acquisition_task_polling(Lsm6dsr *sensor) {
    std::cout << "[Thread] Data acquisition started (POLLING MODE)." << std::endl;

    int sample_count = 0;
    sensor->startFifo();
    auto start_time = std::chrono::steady_clock::now();
    
    const int poll_interval_ms = 10;
    
    while (running) {
        std::vector<ImuData> data = sensor->readFifo();
        
        if (!data.empty()) {
            sample_count += data.size();
            
            // Print every ~2 second (test hz, we get 110hz)
            if (sample_count % 20 < data.size()) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                float rate = (float)sample_count * 1000.0f / elapsed;
                
                // Get the last accelerometer and gyroscope data
                ImuData last_data = {};
                for (auto it = data.rbegin(); it != data.rend(); ++it) {
                    last_data = *it;
                }
                
                if (sensor->is_debug()) std::cout << "[Debug] Samples: " << sample_count 
                          << " | Rate: " << rate << " Hz"
                          << " | XL: " << last_data.ax << ", " << last_data.ay << ", " << last_data.az
                          << " | GY: " << last_data.gx << ", " << last_data.gy << ", " << last_data.gz
                          << std::endl;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    }
    sensor->stopAndClearFifo();
    if (sensor->is_debug()) std::cout << "[Debug] Stopped. Total samples: " << sample_count << std::endl;
}

// Thread function to handle interrupts (GPIO mode)
void data_acquisition_task(Lsm6dsr *sensor, GpioPin *gpio) {
    std::string value_path = gpio->get_value_path();
    int gpio_fd = open(value_path.c_str(), O_RDONLY | O_NONBLOCK);
    
    if (gpio_fd < 0) {
        std::cerr << "[Error] Failed to open GPIO value fd: " << value_path << std::endl;
        return;
    }
    if (sensor->is_debug()) std::cout << "[Debug] GPIO fd opened: " << value_path << std::endl;

    // Setup poll BEFORE restarting sensor
    struct pollfd pfd;
    pfd.fd = gpio_fd;
    pfd.events = POLLPRI | POLLERR;

    // Clear any pending events
    char buf[8];
    lseek(gpio_fd, 0, SEEK_SET);
    read(gpio_fd, buf, sizeof(buf));

    int interrupt_count = 0;
    int poll_timeout_count = 0;

    sensor->startFifo();
    while (running) {
        // Wait for interrupt (timeout 20ms)
        int ret = poll(&pfd, 1, 20);

        if (ret > 0) {
            if (pfd.revents & POLLPRI) {
                // Interrupt occurred: Clear file pointer
                lseek(gpio_fd, 0, SEEK_SET);
                read(gpio_fd, buf, sizeof(buf));

                interrupt_count++;
                
                // Read ALL data from FIFO to let INT1 go LOW
                std::vector<ImuData> data = sensor->readFifo();
                
                // Process Data
                if (sensor->is_debug() && interrupt_count % 20 == 0){
                    if (!data.empty()) {
                        std::cout << "[INT #" << interrupt_count << "] Count: " << data.size();
                        for (const auto& d : data) {
                                std::cout << " | ACC: " << d.ax << ", " << d.ay << ", " << d.az
                                << " | GYRO: " << d.gx << ", " << d.gy << ", " << d.gz << std::endl;
                            }
                        }
                    else {
                        std::cout << "[INT #" << interrupt_count << "] FIFO empty" << std::endl;
                    }
                }
            }
        } else if (ret == 0) {
            // Timeout - print debug info periodically
            poll_timeout_count++;
            lseek(gpio_fd, 0, SEEK_SET);
            int n = read(gpio_fd, buf, sizeof(buf));
            if (n > 0 && buf[0] == '1') {
                if (sensor->is_debug()) {
                    std::cout << "[Debug] GPIO stuck HIGH. Force reading FIFO... ";
                }
                // clean FIFO
                std::vector<ImuData> data = sensor->readFifo();
                if (sensor->is_debug()) std::cout << "[Debug] Have read " << data.size() << " samples." << std::endl;
                    
                if (!data.empty()) {
                    if (sensor->is_debug()) std::cout << "[Debug] Sample: XL " << data[0].ax << ", " << data[0].ay << std::endl;
                } else {
                    if (sensor->is_debug()) std::cout << "[Debug] GPIO is 1 but FIFO Read returned 0 samples!" << std::endl;
                    // 强制重置策略——如果连续读不到，可能需要reset FIFO
                    sensor->stopAndClearFifo();
                    sensor->startFifo();
                }
            }
        } else {
            std::cerr << "[Thread] Poll error: " << strerror(errno) << std::endl;
        }
    }
    
    close(gpio_fd);
    sensor->stopAndClearFifo();
    if (sensor->is_debug()) std::cout << "[Debug] Stopped. Total interrupts: " << interrupt_count << std::endl;
}

int main(int argc, char* argv[]) {
    bool use_polling = false;
    if (argc > 1 && std::string(argv[1]) == "-p") {
        use_polling = true;
        std::cout << "=== POLLING MODE ===" << std::endl;
    } else {
        std::cout << "=== GPIO INTERRUPT MODE ===" << std::endl;
    }

    Lsm6dsr lsm(I2C_BUS_ID, LSM_ADDR, true);
    if (!lsm.init(ACC_ODR_104 | ACC_4g, GYRO_ODR_104 | GYRO_250dps, 0x44)) {
        std::cerr << "[Error] Lsm6dsr initialization failed!" << std::endl;
        return -1;
    }

    if (use_polling) {
        // Polling mode - no GPIO needed
        std::thread acquisition_thread(data_acquisition_task_polling, &lsm);
        
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.get();

        running = false;
        acquisition_thread.join();
    } else {
        // GPIO interrupt mode
        GpioPin int1(INT1_GPIO_PIN, GpioPin::DIRECTION::IN, GpioPin::EDGE::RISING);
        std::cout << "[INFO] GPIO " << INT1_GPIO_PIN << " configured as input with RISING edge" << std::endl;
        if (lsm.is_debug()) std::cout << "[Debug] GPIO initial value: " << int1.read() << std::endl;
        
        std::thread acquisition_thread(data_acquisition_task, &lsm, &int1);

        std::cout << "[INFO] Press Enter to exit..." << std::endl;
        std::cin.get();

        running = false;
        acquisition_thread.join();
    }
    // 挂起节能
    lsm.reset();
    return 0;
}