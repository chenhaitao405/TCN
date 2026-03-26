#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include "peripherals/bm85163.h"

#define I2C_BUS_ID 2

std::atomic<bool> running(true);

void rtc_monitor_task(Bm85163 *rtc) {
    if (rtc->isDebug()) std::cout << "[Debug] RTC monitor started." << std::endl;

    while (running) {
        RtcTime t = rtc->getTime();
        
        // 格式化输出 2025-01-01 12:00:43 [Weekday: 3]
        std::cout << "\r[Time] " 
                  << t.year << "-" 
                  << std::setw(2) << std::setfill('0') << (int)t.month << "-"
                  << std::setw(2) << std::setfill('0') << (int)t.day << " "
                  << std::setw(2) << std::setfill('0') << (int)t.hour << ":"
                  << std::setw(2) << std::setfill('0') << (int)t.minute << ":"
                  << std::setw(2) << std::setfill('0') << (int)t.second
                  << " [Weekday: " << (int)t.weekday << "]"
                  << (t.vl_valid ? "" : " [LOW VOLTAGE WARNING]")
                  << std::flush;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "      BM85163 RTC Test Program" << std::endl;
    std::cout << "================================================" << std::endl;

    Bm85163 rtc(I2C_BUS_ID, BM85163_ADDR, true);

    if (!rtc.init()) {
        std::cerr << "[Error] BM85163 initialization failed!" << std::endl;
        std::cerr << "Please check I2C connection and address (0x51)." << std::endl;
        return -1;
    }

    // 询问是否校准时间
    std::cout << "Do you want to set the time to Compile Time (2025-01-01 12:00:00)? (y/n): ";
    char c;
    std::cin >> c;
    if (c == 'y' || c == 'Y') {
        RtcTime newTime;
        newTime.year = 2025;
        newTime.month = 1;
        newTime.day = 1;
        newTime.weekday = 3; // Wednesday
        newTime.hour = 12;
        newTime.minute = 0;
        newTime.second = 0;
        
        if (rtc.setTime(newTime)) {
            std::cout << "[Info] Time updated successfully." << std::endl;
        } else {
            std::cerr << "[Error] Failed to set time." << std::endl;
        }
    }
    
    // 清空输入缓冲区以免影响后续
    std::cin.ignore(1000, '\n'); 

    std::cout << "[Info] Starting background monitoring thread..." << std::endl;
    std::thread monitor_thread(rtc_monitor_task, &rtc);

    std::cout << "\n>>> Press ENTER to stop <<<\n" << std::endl;
    std::cin.get();

    running = false;
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }

    std::cout << "Done." << std::endl;
    return 0;
}