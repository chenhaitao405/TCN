#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <iomanip>
#include "qmc5883p.h"

#define I2C_BUS_ID      2
#define QMC5883P_ADDR   0x2C

std::atomic<bool> running(true);

void mag_acquisition_task_polling(Qmc5883p *sensor) {
    if (sensor->isDebug()) std::cout << "[Debug] QMC5883P acquisition started (POLLING MODE)." << std::endl;

    int sample_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    const int poll_interval_ms = 9; 
    
    while (running) {
        if (sensor->isDataReady()) {
            MagData data = sensor->readData();
            sample_count++;
            
            if (sample_count % 20 == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                float rate = (elapsed > 0) ? (float)sample_count * 1000.0f / elapsed : 0;
                
                if (sensor->isDebug()) std::cout << std::fixed << std::setprecision(2)
                          << "[Debug] Samples: " << sample_count 
                          << " | Rate: " << rate << " Hz"
                          << " | X: " << std::setw(8) << data.mx 
                          << " | Y: " << std::setw(8) << data.my 
                          << " | Z: " << std::setw(8) << data.mz 
                          << std::endl;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    }

    std::cout << "[Thread] Stopped. Total samples collected: " << sample_count << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "   QMC5883P Threaded Polling Test Program" << std::endl;
    std::cout << "================================================" << std::endl;


    Qmc5883p mag(I2C_BUS_ID, QMC5883P_ADDR, true);

    // 配置：NORMAL 模式, 100Hz 频率, ±8G 量程
    if (!mag.init(QMC5883P_MODE_NORMAL, QMC5883P_ODR_100HZ, QMC5883P_RNG_8G, QMC5883P_OSR1_2, QMC5883P_OSR2_2)) {
        std::cerr << "[Error] QMC5883P initialization failed!" << std::endl;
        std::cerr << "Please check I2C connection and address (0x2C)." << std::endl;
        return -1;
    }

    std::cout << "[Info] Sensor initialized. Starting background thread..." << std::endl;

    std::thread acquisition_thread(mag_acquisition_task_polling, &mag);

    std::cout << "\n>>> Press ENTER to stop data acquisition <<<\n" << std::endl;
    std::cin.get();

    running = false;
    if (acquisition_thread.joinable()) {
        acquisition_thread.join();
    }

    // 传感器挂起 Suspend Mode，节能
    mag.writeReg(0x0A, 0x00);

    std::cout << "Done." << std::endl;
    return 0;
}