#include "peripherals/qmc5883p.h"
#include <iostream>
#include <thread>
#include <chrono>

Qmc5883p::Qmc5883p(int bus, uint8_t addr, bool debug) 
    : I2cDevice(bus, addr), _debug(debug), _sensitivity(QMC5883P_SENS_8G) {}

bool Qmc5883p::softReset() {
    return writeReg(QMC5883P_REG_CONTROL2, QMC5883P_SOFT_RESET);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

bool Qmc5883p::init(uint8_t mode, uint8_t odr, uint8_t rng, uint8_t osr1, uint8_t osr2) {
    uint8_t id;
    if (!readReg(QMC5883P_REG_CHIP_ID, id) || id != QMC5883P_ID) {
        std::cerr << "ID Error: Got 0x" << std::hex << (int)id << " expected 0x80" << std::endl;
        return false;
    }

    softReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // 更新灵敏度
    if (rng == QMC5883P_RNG_2G) _sensitivity = QMC5883P_SENS_2G;
    else if (rng == QMC5883P_RNG_8G) _sensitivity = QMC5883P_SENS_8G;
    else if (rng == QMC5883P_RNG_12G) _sensitivity = QMC5883P_SENS_12G;
    else _sensitivity = QMC5883P_SENS_30G;

    // 配置 CONTROL 2
    writeReg(QMC5883P_REG_CONTROL2, rng);

    // 配置 CONTROL 1
    writeReg(QMC5883P_REG_CONTROL1, mode | odr | osr1 | osr2);

    return true;
}

bool Qmc5883p::isDataReady() {
    uint8_t status;
    readReg(QMC5883P_REG_STATUS, status);
    return (status & QMC5883P_STATUS_DRDY) != 0;
}

MagData Qmc5883p::readData() {
    uint8_t buf[6];
    MagData data = {0};
    
    // 读取磁场数据
    if (readRegs(QMC5883P_REG_XOUT_L, buf, 6)) {
        int16_t x = (int16_t)(buf[0] | (buf[1] << 8));
        int16_t y = (int16_t)(buf[2] | (buf[3] << 8));
        int16_t z = (int16_t)(buf[4] | (buf[5] << 8));
        int16_t t = (int16_t)(buf[6] | (buf[7] << 8));

        data.mx = x * _sensitivity;
        data.my = y * _sensitivity;
        data.mz = z * _sensitivity;
    }
    return data;
}