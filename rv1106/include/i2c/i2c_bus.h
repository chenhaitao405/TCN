#ifndef I2C_BUS_H
#define I2C_BUS_H

#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <cstddef>
#include <cstdint>

class I2cDevice {
protected:
    int fd;
    int bus;
    uint8_t address;
    bool connected;

public:
    explicit I2cDevice(int bus_num, uint8_t dev_addr);
    virtual ~I2cDevice();

    bool writeReg(uint8_t reg, uint8_t value);
    bool readReg(uint8_t reg, uint8_t &value);
    bool readRegs(uint8_t reg, uint8_t *buffer, size_t length);

    bool isConnected() const;
};

#endif