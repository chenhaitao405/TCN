#include "bm85163.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>

Bm85163::Bm85163(int bus, uint8_t addr, bool debug)
    : I2cDevice(bus, addr), _debug(debug) {}

uint8_t Bm85163::bcd2dec(uint8_t val) {
    return ((val >> 4) * 10) + (val & 0x0F);
}

uint8_t Bm85163::dec2bcd(uint8_t val) {
    return ((val / 10) << 4) + (val % 10);
}

bool Bm85163::init() {
    uint8_t val;
    // 尝试读取 Control 1 寄存器来验证连接
    if (!readReg(BM85163_REG_CONTROL1, val)) {
        std::cerr << "[BM85163] Init Failed: I2C read error." << std::endl;
        return false;
    }

    // 检查 Voltage Low 标志
    if (isVoltageLow()) {
        if (_debug) std::cout << "[BM85163] Warning: Voltage Low flag set. Data may be invalid." << std::endl;
        // 注意：VL 位必须通过软件清除，通常是在设置新时间时清除
    }

    // 确保 STOP 位为 0，让时钟运行
    if (val & BM85163_BIT_STOP) {
        if (_debug) std::cout << "[BM85163] Clearing STOP bit to start clock." << std::endl;
        val &= ~BM85163_BIT_STOP;
        writeReg(BM85163_REG_CONTROL1, val);
    }

    return true;
}

bool Bm85163::isVoltageLow() {
    uint8_t sec_reg;
    if (readReg(BM85163_REG_SECONDS, sec_reg)) {
        return (sec_reg & BM85163_BIT_VL) != 0;
    }
    return true;
}

RtcTime Bm85163::getTime() {
    RtcTime t = {0};
    uint8_t buf[7]; // 从 0x02 (Seconds) 到 0x08 (Years) 共 7 个字节

    // 必须一次性读取所有时间寄存器，以防止进位错误 (见 datasheet Figure 9)
    if (readRegs(BM85163_REG_SECONDS, buf, 7)) {
        // 0x02: Seconds (bit 7 is VL)
        t.vl_valid = (buf[0] & BM85163_BIT_VL) == 0;
        t.second = bcd2dec(buf[0] & 0x7F);

        // 0x03: Minutes
        t.minute = bcd2dec(buf[1] & 0x7F);

        // 0x04: Hours (bit 5-4 tens, 3-0 units)
        t.hour = bcd2dec(buf[2] & 0x3F);

        // 0x05: Days
        t.day = bcd2dec(buf[3] & 0x3F);

        // 0x06: Weekdays (0-6)
        t.weekday = buf[4] & 0x07;

        // 0x07: Century_months (Bit 7 is Century flag, Bit 4-0 Month)
        // Century 位在年份从 99->00 溢出时翻转。通常假设 C=0 为 20xx。
        bool century_bit = (buf[5] & 0x80) >> 7;
        t.month = bcd2dec(buf[5] & 0x1F);

        // 0x08: Years
        t.year = 2000 + bcd2dec(buf[6]); 
        // 如果需要处理 19xx/20xx，可以根据 century_bit 调整逻辑，这里简化为 2000+
    } else {
        std::cerr << "[BM85163] Read time failed." << std::endl;
    }
    return t;
}

bool Bm85163::setTime(const RtcTime &t) {
    uint8_t buf[7];

    // 设置 STOP 位 (Control 1 reg bit 5) 以确保精确写入 (Datasheet 6.10)
    uint8_t ctrl1;
    readReg(BM85163_REG_CONTROL1, ctrl1);
    writeReg(BM85163_REG_CONTROL1, ctrl1 | BM85163_BIT_STOP);

    // 准备数据
    // Seconds: 确保 VL 位被清除 (0)
    buf[0] = dec2bcd(t.second) & 0x7F; 
    buf[1] = dec2bcd(t.minute);
    buf[2] = dec2bcd(t.hour);
    buf[3] = dec2bcd(t.day);
    buf[4] = t.weekday & 0x07;
    // Month: 设置当前世纪位 (这里假设是 20xx，视作 Century 0 或 1，通常保持读取时的状态或重置)
    // 简单起见，写入时 Century Bit 设为 0
    buf[5] = dec2bcd(t.month); 
    buf[6] = dec2bcd(t.year % 100);

    // 写入时间寄存器 (一次性写入以确保一致性)
    // 注意：writeRegs 的实现依赖于底层 I2cDevice。如果不支持一次写多个，需改为单字节写。
    // BM85163 支持地址自动递增。
    bool success = true;
    for (int i = 0; i < 7; i++) {
        if (!writeReg(BM85163_REG_SECONDS + i, buf[i])) {
            success = false;
            break;
        }
    }

    // 清除 STOP 位，让 RTC 开始走时
    writeReg(BM85163_REG_CONTROL1, ctrl1 & ~BM85163_BIT_STOP);
    
    if (_debug) std::cout << "[BM85163] Time set complete." << std::endl;
    
    return success;
}