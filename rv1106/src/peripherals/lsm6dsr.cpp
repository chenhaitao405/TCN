#include "peripherals/lsm6dsr.h"
#include <cmath>
#include <thread>
#include <chrono>

Lsm6dsr::Lsm6dsr(int bus, uint8_t addr, bool debug, bool fifo_mode) : I2cDevice(bus, addr), _debug(debug), _fifo_mode(fifo_mode){
    
}

Lsm6dsr::~Lsm6dsr() noexcept {
    if (!isConnected()) {
        return;
    }
    stop();
    reset();
}

std::unique_ptr<Lsm6dsr> Lsm6dsr::create(int bus, uint8_t addr,
                                         uint8_t AccFlag, uint8_t GyroFlag,
                                         bool debug) {
    auto sensor = std::unique_ptr<Lsm6dsr>(new Lsm6dsr(bus, addr, debug, false));
    if (!sensor->init(AccFlag, GyroFlag)) {
        return nullptr;
    }
    return sensor;
}

std::unique_ptr<Lsm6dsr> Lsm6dsr::create(int bus, uint8_t addr,
                                         uint8_t AccFlag, uint8_t GyroFlag,
                                         uint8_t FifoRate,
                                         bool debug, bool fifo_mode) {
    auto sensor = std::unique_ptr<Lsm6dsr>(new Lsm6dsr(bus, addr, debug, fifo_mode));
    if (!sensor->init(AccFlag, GyroFlag, FifoRate)) {
        return nullptr;
    }
    return sensor;
}

bool Lsm6dsr::reset() {
    if (!isConnected() || !this->checkId()) return false;
    // 复位
    writeReg(CTRL3_C, 0x01);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return true;
}

bool Lsm6dsr::init(uint8_t AccFlag, uint8_t GyroFlag, uint8_t FifoRate) {
    if (!isConnected() || !this->checkId()) return false;
    reset();  
    // ⭐start BDU ⭐IF_INC必须为1
    writeReg(CTRL3_C, 0x44);
    // start timestamp record
    writeReg(CTRL10_C, 0x20);
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

bool Lsm6dsr::init(uint8_t AccFlag, uint8_t GyroFlag) {
    if (!isConnected() || !this->checkId()) return false;
    reset();
    // ⭐start BDU ⭐IF_INC必须为1
    writeReg(CTRL3_C, 0x44);
    // start timestamp record
    writeReg(CTRL10_C, 0x20);
    startAccelerometer(AccFlag);
    startGyroscope(GyroFlag);
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
    if (!readReg(FIFO_STATUS1, status1) || !readReg(FIFO_STATUS2, status2)) {
        std::cerr << "[Error] Failed to read FIFO status!" << std::endl;
        return dataList;
    }
    
    uint16_t count = status1 | ((status2 & 0x03) << 8);
    if (count == 0) return dataList;
    static uint32_t last_timestamp = 0;
    static bool acc_valid = false, gyro_vaild = false;
    static ImuData data = {0};

    for (uint16_t i = 0; i < count; i++) {
        uint8_t raw[7];
        if (readRegs(FIFO_DATA_OUT_TAG, raw, 7)) {
            uint8_t tag = raw[0] >> 3;
            int16_t x_raw = (int16_t)(raw[1] | (raw[2] << 8));
            int16_t y_raw = (int16_t)(raw[3] | (raw[4] << 8));
            int16_t z_raw = (int16_t)(raw[5] | (raw[6] << 8));

            if (tag == 0x02) { // XL
                data.acc_x = (x_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
                data.acc_y = (y_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
                data.acc_z = (z_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
                acc_valid = true;
            } else if (tag == 0x01) { // GY
                data.gyro_x = (x_raw * SENSITIVITY_G_250) / 1000.0f;
                data.gyro_y = (y_raw * SENSITIVITY_G_250) / 1000.0f;
                data.gyro_z = (z_raw * SENSITIVITY_G_250) / 1000.0f;
                gyro_vaild = true;
            } else if(tag == 0x04) {
                last_timestamp = (uint32_t)(raw[1] | (raw[2] << 8) | (raw[3] << 16) | (raw[4] << 24));
            }
        }

        if (acc_valid && gyro_vaild) {
            data.timestamp = last_timestamp;
            dataList.push_back(data);
            acc_valid = false;
            gyro_vaild = false;
        }
    }
    return dataList;
}

bool Lsm6dsr::readImuData(ImuData &imu_data, uint8_t &is_new_data) {
    uint8_t raw[12];
    if (readRegs(OUTX_L_G, raw, 12)) {
        int16_t gryo_x_raw = (int16_t)(raw[0] | (raw[1] << 8));
        int16_t gryo_y_raw = (int16_t)(raw[2] | (raw[3] << 8));
        int16_t gryo_z_raw = (int16_t)(raw[4] | (raw[5] << 8));
        int16_t acc_x_raw = (int16_t)(raw[6] | (raw[7] << 8));
        int16_t acc_y_raw = (int16_t)(raw[8] | (raw[9] << 8));
        int16_t acc_z_raw = (int16_t)(raw[10] | (raw[11] << 8));
        imu_data.acc_x = (acc_x_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
        imu_data.acc_y = (acc_y_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
        imu_data.acc_z = (acc_z_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
        imu_data.gyro_x = (gryo_x_raw * SENSITIVITY_G_250) / 1000.0f;
        imu_data.gyro_y = (gryo_y_raw * SENSITIVITY_G_250) / 1000.0f;
        imu_data.gyro_z = (gryo_z_raw * SENSITIVITY_G_250) / 1000.0f;
    } else return false;

    if (readReg(STATUS_REG, is_new_data)) {
    } else return false;

    uint8_t ts_raw[4];
    if (readRegs(TIMESTAMP0, ts_raw, 4)) {
        // ⭐这里的 timestamp单位是25us
        // 如果需要转换为秒或毫秒，可以手动换算：time_ms = imu_data.timestamp * 0.025f;
        imu_data.timestamp = (uint32_t)(ts_raw[0] | (ts_raw[1] << 8) | (ts_raw[2] << 16) | (ts_raw[3] << 24));
    } else {
        return false;
    }
    return true;
}

bool Lsm6dsr::readImuData(ImuData &imu_data) {
    uint8_t raw[12];
    if (readRegs(OUTX_L_G, raw, 12)) {
        int16_t gryo_x_raw = (int16_t)(raw[0] | (raw[1] << 8));
        int16_t gryo_y_raw = (int16_t)(raw[2] | (raw[3] << 8));
        int16_t gryo_z_raw = (int16_t)(raw[4] | (raw[5] << 8));
        int16_t acc_x_raw = (int16_t)(raw[6] | (raw[7] << 8));
        int16_t acc_y_raw = (int16_t)(raw[8] | (raw[9] << 8));
        int16_t acc_z_raw = (int16_t)(raw[10] | (raw[11] << 8));
        imu_data.acc_x = (acc_x_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
        imu_data.acc_y = (acc_y_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
        imu_data.acc_z = (acc_z_raw * SENSITIVITY_XL_4G) / 1000.0f * 9.81f;
        imu_data.gyro_x = (gryo_x_raw * SENSITIVITY_G_250) / 1000.0f;
        imu_data.gyro_y = (gryo_y_raw * SENSITIVITY_G_250) / 1000.0f;
        imu_data.gyro_z = (gryo_z_raw * SENSITIVITY_G_250) / 1000.0f;
    } else return false;

    uint8_t ts_raw[4];
    if (readRegs(TIMESTAMP0, ts_raw, 4)) {
        // ⭐这里的 timestamp单位是25us
        // 如果需要转换为秒或毫秒，可以手动换算：time_ms = imu_data.timestamp * 0.025f;
        imu_data.timestamp = (uint32_t)(ts_raw[0] | (ts_raw[1] << 8) | (ts_raw[2] << 16) | (ts_raw[3] << 24));
    } else {
        return false;
    }
    return true;
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

void Lsm6dsr::stop() {
    stopGyroscope();
    stopAccelerometer();
    if (this->_fifo_mode) {
        stopAndClearFifo();
    }
    // stop timestamp counter
    writeReg(CTRL10_C, 0x00);
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
    writeReg(FIFO_CTRL4, 0x46);

    if (this->_debug) {
        std::cout << "[Debug] LSM6DSR FIFO started. INT1 will go HIGH when data ready." << std::endl;
        printFifoStatus();
    }
}