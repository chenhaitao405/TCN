#include "i2c/i2c_bus.h"

I2cDevice::I2cDevice(int bus_num, uint8_t dev_addr)
    : fd(-1), bus(bus_num), address(dev_addr), connected(false) {
    std::string filename = "/dev/i2c-" + std::to_string(bus);
    fd = open(filename.c_str(), O_RDWR);
    if (fd < 0) {
        std::cerr << "[I2C] Failed to open bus " << bus << std::endl;
        return;
    }

    if (ioctl(fd, I2C_SLAVE, address) < 0) {
        std::cerr << "[I2C] Failed to acquire bus access and/or talk to slave" << std::endl;
        close(fd);
        fd = -1;
        return;
    }
    connected = true;
}

I2cDevice::~I2cDevice() {
    if (fd >= 0) {
        close(fd);
    }
}

bool I2cDevice::writeReg(uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    if (write(fd, buf, 2) != 2) {
        std::cerr << "[I2C] Write failed to reg 0x" << std::hex << static_cast<int>(reg) << std::dec << std::endl;
        return false;
    }
    return true;
}

bool I2cDevice::readReg(uint8_t reg, uint8_t &value) {
    if (write(fd, &reg, 1) != 1) return false;
    if (read(fd, &value, 1) != 1) return false;
    return true;
}

bool I2cDevice::readRegs(uint8_t reg, uint8_t *buffer, size_t length) {
    if (write(fd, &reg, 1) != 1) return false;
    if (read(fd, buffer, length) != static_cast<ssize_t>(length)) return false;
    return true;
}

bool I2cDevice::isConnected() const {
    return connected;
}
