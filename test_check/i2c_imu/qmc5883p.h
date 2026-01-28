#pragma once
#include <stdint.h>
#include <vector>
#include "i2c_bus.h"

// QMC5883P 默认 I2C 地址
#define QMC5883P_ADDR 0x2C

// 寄存器映射
#define QMC5883P_REG_CHIP_ID 0x00 // 芯片 ID 寄存器
#define QMC5883P_REG_XOUT_L 0x01
#define QMC5883P_REG_XOUT_H 0x02
#define QMC5883P_REG_YOUT_L 0x03
#define QMC5883P_REG_YOUT_H 0x04
#define QMC5883P_REG_ZOUT_L 0x05
#define QMC5883P_REG_ZOUT_H 0x06
#define QMC5883P_REG_TEMP_L 0x07
#define QMC5883P_REG_TEMP_H 0x08
#define QMC5883P_REG_STATUS 0x09   // 状态寄存器
#define QMC5883P_REG_CONTROL1 0x0A // 控制寄存器 1 (Mode, ODR, OSR2, OSR1)
#define QMC5883P_REG_CONTROL2 0x0B // 控制寄存器 2 (Soft Reset, Range)

// 芯片 ID 值
#define QMC5883P_ID 0x80

// STATUS 寄存器位定义
#define QMC5883P_STATUS_DRDY 0x01 // 数据就绪位 (bit 0)
#define QMC5883P_STATUS_OVFL 0x02 // 溢出位 (bit 1)

// CONTROL 1 寄存器位定义 (0x0A)
#define QMC5883P_MODE_SUSPEND 0x00
#define QMC5883P_MODE_NORMAL 0x01
#define QMC5883P_MODE_SINGLE 0x02
#define QMC5883P_MODE_CONTINUOUS 0x03
#define QMC5883P_ODR_10HZ (0x00 << 2)
#define QMC5883P_ODR_50HZ (0x01 << 2)
#define QMC5883P_ODR_100HZ (0x02 << 2)
#define QMC5883P_ODR_200HZ (0x03 << 2)

// CONTROL 2 寄存器位定义 (0x0B)
#define QMC5883P_SOFT_RESET 0x80
#define QMC5883P_RNG_2G (0x03 << 2)
#define QMC5883P_RNG_8G (0x02 << 2)
#define QMC5883P_RNG_12G (0x01 << 2)
#define QMC5883P_RNG_30G (0x00 << 2)

// 灵敏度换算系数 (mG/LSB = 1000 / LSB/G)
#define QMC5883P_SENS_2G (1.0f / 15.0f) // 15000 LSB/G
#define QMC5883P_SENS_8G (1.0f / 3.75f) // 3750 LSB/G
#define QMC5883P_SENS_12G (1.0f / 2.5f) // 2500 LSB/G
#define QMC5883P_SENS_30G 1.0f          // 1000 LSB/G

// OSR1系数
// Over sample Rate (OSR1) registers are used to control bandwidth of an internal digital filter.
#define QMC5883P_OSR1_8 0x00
#define QMC5883P_OSR1_4 0x10
#define QMC5883P_OSR1_2 0x20
#define QMC5883P_OSR1_1 0x30

// OSR2系数
// OSR2 control another filter for better noise performance
#define QMC5883P_OSR2_8 0xC0
#define QMC5883P_OSR2_4 0x80
#define QMC5883P_OSR2_2 0x40
#define QMC5883P_OSR2_1 0x00

struct MagData
{
    float mx, my, mz; // mGauss
    uint32_t timestamp;
};

class Qmc5883p : public I2cDevice
{
public:
    explicit Qmc5883p(int bus, uint8_t addr = QMC5883P_ADDR, bool debug = false);
    bool init(uint8_t mode = QMC5883P_MODE_NORMAL, uint8_t odr = QMC5883P_ODR_100HZ, uint8_t rng = QMC5883P_RNG_8G,
              uint8_t osr1 = QMC5883P_OSR1_4, uint8_t osr2 = QMC5883P_OSR2_1);
    MagData readData();
    bool isDataReady();
    bool softReset();
    bool isDebug() { return _debug; }

private:
    float _sensitivity;
    uint8_t _range_bits;
    bool _debug;
};