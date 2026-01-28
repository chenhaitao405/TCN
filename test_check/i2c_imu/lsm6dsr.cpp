#include "lsm6dsr.h"
#include <cmath>
#include <thread>
#include <chrono>

Lsm6dsr::Lsm6dsr(int bus, uint8_t addr, bool debug) : I2cDevice(bus, addr), _debug(debug){
    
}

bool Lsm6dsr::reset() {
    if (!isConnected() || !this->checkId()) return false;
    // 复位
    writeReg(CTRL3_C, 0x01);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return true;
}

bool Lsm6dsr::init(uint8_t AccFlag, uint8_t GyroFlag, uint8_t FifoRate) {
    if (!isConnected() || !this->checkId()) return false;
    // reset();  
    // start BDU ⭐必须设置为open-drain⭐
    writeReg(CTRL3_C, 0x44);

    startAccelerometer(AccFlag);
    startGyroscope(GyroFlag);

    // set watermask
    writeReg(FIFO_CTRL1, 0x02);
    writeReg(FIFO_CTRL2, 0x00);

    writeReg(FIFO_CTRL3, FifoRate);

    stopAndClearFifo();
    
    if (_debug) std::cout << "[Debug] LSM6DSR Init done. FIFO Reset. When host interupt is ready, then start FIFO!" << std::endl;
    return true;
}

bool Lsm6dsr::checkId() {
    uint8_t who_am_i;
    readReg(WHO_AM_I, who_am_i);
    if (who_am_i != 0x6B) {
        std::cerr << "[Error] LSM6DSR ID mismatch!" << std::endl;
        return false;
    }
    return true;
}

void Lsm6dsr::clearInterrupt() {
    uint8_t status;
    readReg(ALL_INT_SRC, status);
}

std::vector<ImuData> Lsm6dsr::readFifo() {
    std::vector<ImuData> dataList;
    uint8_t status1 = 0, status2 = 0;
    uint8_t raw[7];
    if (!readReg(FIFO_STATUS1, status1) || !readReg(FIFO_STATUS2, status2)) {
        std::cerr << "[Error] Failed to read FIFO status!" << std::endl;
        return dataList; 
    }

    uint16_t count = status1 | ((status2 & 0x03) << 8);

    if (count == 0) return dataList;

    uint16_t i = 0;
    for (int i = 0; i < count / 2; i++) {
        bool acc_valid = false, gyro_vaild = false;
        ImuData data = {0};
        if (readRegs(FIFO_DATA_OUT_TAG, raw, 7)) {
            uint8_t tag = raw[0] >> 3;
        
            int16_t x_raw = (int16_t)(raw[1] | (raw[2] << 8));
            int16_t y_raw = (int16_t)(raw[3] | (raw[4] << 8));
            int16_t z_raw = (int16_t)(raw[5] | (raw[6] << 8));

            if (tag == 0x02) { // XL
                data.ax = (x_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
                data.ay = (y_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
                data.az = (z_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
                acc_valid = true;
            } else if (tag == 0x01) { // GY
                data.gx = (x_raw * SENSITIVITY_G_250) / 1000.0f;
                data.gy = (y_raw * SENSITIVITY_G_250) / 1000.0f;
                data.gz = (z_raw * SENSITIVITY_G_250) / 1000.0f;
                gyro_vaild = true;
            }
        }

        if (readRegs(FIFO_DATA_OUT_TAG, raw, 7)) {
            uint8_t tag = raw[0] >> 3;
            
            int16_t x_raw = (int16_t)(raw[1] | (raw[2] << 8));
            int16_t y_raw = (int16_t)(raw[3] | (raw[4] << 8));
            int16_t z_raw = (int16_t)(raw[5] | (raw[6] << 8));            

            if (tag == 0x02) { // XL
                data.ax = (x_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
                data.ay = (y_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
                data.az = (z_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
                acc_valid = true;
            } else if (tag == 0x01) { // GY
                data.gx = (x_raw * SENSITIVITY_G_250) / 1000.0f;
                data.gy = (y_raw * SENSITIVITY_G_250) / 1000.0f;
                data.gz = (z_raw * SENSITIVITY_G_250) / 1000.0f;
                gyro_vaild = true;
            }
        }
        if (acc_valid && gyro_vaild) {
            dataList.push_back(data);
        }
    }
    return dataList;
}

void Lsm6dsr::printFifoStatus() {
    uint8_t status1, status2, int1_ctrl, all_int_src;
    readReg(FIFO_STATUS1, status1);
    readReg(FIFO_STATUS2, status2);
    readReg(INT1_CTRL, int1_ctrl);
    readReg(ALL_INT_SRC, all_int_src);
    
    uint16_t count = status1 | ((status2 & 0x03) << 8);
    bool fifo_full = (status2 >> 5) & 0x01;
    bool fifo_ovr = (status2 >> 6) & 0x01;
    bool fifo_wtm = (status2 >> 7) & 0x01;
    
    std::cout << "[Debug] FIFO Count: " << count 
              << ", WTM_FLAG: " << fifo_wtm
              << ", FULL: " << fifo_full 
              << ", OVR: " << fifo_ovr 
              << ", INT1_CTRL: 0x" << std::hex << (int)int1_ctrl 
              << ", ALL_INT_SRC: 0x" << (int)all_int_src << std::dec << std::endl;
}

void Lsm6dsr::stopAndClearFifo() {
    // Set FIFO to Bypass mode
    writeReg(FIFO_CTRL4, 0x00);
    // INT1_FIFO_TH Disable
    writeReg(INT1_CTRL, 0x00); 
    
    if (this->_debug){
        std::cout << "[Debug] LSM6DSR FIFO cleared. INT1 should be LOW now." << std::endl;
        printFifoStatus();
    }
}

void Lsm6dsr::stopAccelerometer() {
    writeReg(CTRL1_XL, ACC_POWER_DOWN);  // XL power down
}

void Lsm6dsr::stopGyroscope() {
    writeReg(CTRL2_G, GYRO_POWER_DOWN);   // GY power down
}

void Lsm6dsr::startAccelerometer(uint8_t flag) {
    writeReg(CTRL1_XL, flag);
}

void Lsm6dsr::startGyroscope(uint8_t flag) {
    writeReg(CTRL2_G, flag);
}

void Lsm6dsr::startFifo() {
    // INT1_FIFO_TH Enable
    writeReg(INT1_CTRL, 0x08); 
    // Configure FIFO for continuous mode
    writeReg(FIFO_CTRL4, 0x06);

    if (this->_debug) {
        std::cout << "[Debug] LSM6DSR FIFO started. INT1 will go HIGH when data ready." << std::endl;
        printFifoStatus();
    }
}