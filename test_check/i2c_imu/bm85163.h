#pragma once
#include <stdint.h>
#include <vector>
#include <string>
#include "i2c_bus.h"

#define BM85163_ADDR 0x51

// 寄存器映射
#define BM85163_REG_CONTROL1     0x00 // Control_status_1
#define BM85163_REG_CONTROL2     0x01 // Control_status_2
#define BM85163_REG_SECONDS      0x02 // VL_seconds
#define BM85163_REG_MINUTES      0x03
#define BM85163_REG_HOURS        0x04
#define BM85163_REG_DAYS         0x05 // Day of month (1-31)
#define BM85163_REG_WEEKDAYS     0x06 // Day of week (0-6)
#define BM85163_REG_MONTHS       0x07 // Century_months
#define BM85163_REG_YEARS        0x08
#define BM85163_REG_ALARM_MIN    0x09
#define BM85163_REG_CLKOUT_CTL   0x0D
#define BM85163_REG_TIMER_CTL    0x0E
#define BM85163_REG_TIMER        0x0F

// CONTROL 1 寄存器位定义 (0x00)
#define BM85163_BIT_TEST1        0x80
#define BM85163_BIT_STOP         0x20 // 1: Stop RTC source clock, 0: Run
#define BM85163_BIT_TESTC        0x08

// VL_SECONDS 寄存器位定义 (0x02)
#define BM85163_BIT_VL           0x80 // Voltage Low flag

// 时间数据结构
struct RtcTime {
    uint16_t year;   // e.g., 2023
    uint8_t month;   // 1-12
    uint8_t day;     // 1-31
    uint8_t weekday; // 0-6 (Sun-Sat)
    uint8_t hour;    // 0-23
    uint8_t minute;  // 0-59
    uint8_t second;  // 0-59
    bool vl_valid;   // True if Voltage Low flag was NOT set (time is trusted)
};

class Bm85163 : public I2cDevice {
public:
    explicit Bm85163(int bus, uint8_t addr = BM85163_ADDR, bool debug = false);

    bool init();

    RtcTime getTime();

    bool setTime(const RtcTime &time);

    // 检查是否有低电压警告 (意味着时间可能不准确)
    bool isVoltageLow();

    bool isDebug() const { return _debug; }

private:
    bool _debug;

    // BCD 转换辅助函数
    uint8_t bcd2dec(uint8_t val);
    uint8_t dec2bcd(uint8_t val);
};