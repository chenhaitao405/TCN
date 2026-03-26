#ifndef I2C_BUS_H
#define I2C_BUS_H

#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <cstdint>
#include <vector>
#include <cstring>

class I2cDevice {
protected:
    int fd;
    int bus;
    uint8_t address;
    bool connected;

public:
    explicit I2cDevice(int bus_num, uint8_t dev_addr) : bus(bus_num), address(dev_addr), connected(false) {
        std::string filename = "/dev/i2c-" + std::to_string(bus);
        fd = open(filename.c_str(), O_RDWR);
        if (fd < 0) {
            std::cerr << "[I2C] Failed to open bus " << bus << std::endl;
            return;
        }

        if (ioctl(fd, I2C_SLAVE, address) < 0) {
            std::cerr << "[I2C] Failed to acquire bus access and/or talk to slave" << std::endl;
            return;
        }
        connected = true;
    }

    virtual ~I2cDevice() {
        if (fd >= 0) {
            close(fd);
        }
    }

    bool writeReg(uint8_t reg, uint8_t value) {
        uint8_t buf[2] = {reg, value};
        if (write(fd, buf, 2) != 2) {
            std::cerr << "[I2C] Write failed to reg 0x" << std::hex << (int)reg << std::dec << std::endl;
            return false;
        }
        return true;
    }

    bool readReg(uint8_t reg, uint8_t &value) {
        if (write(fd, &reg, 1) != 1) return false;
        if (read(fd, &value, 1) != 1) return false;
        return true;
    }

    // Burst read for FIFO
    bool readRegs(uint8_t reg, uint8_t *buffer, size_t length) {
        if (write(fd, &reg, 1) != 1) return false;
        if (read(fd, buffer, length) != length) return false;
        return true;
    }

    bool isConnected() const { return connected; }
};

#endif