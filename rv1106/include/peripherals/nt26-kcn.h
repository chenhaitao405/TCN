#pragma once

#include "serial/uart_serial.h"
#include <memory>
#define NT26_PWRON_PIN 40
#define NT26_UART_TTY "/dev/ttyS1"

class NT26KCN : public SerialBase
{
protected:
    explicit NT26KCN(int fd) : SerialBase(fd) {}

public:
    static std::unique_ptr<NT26KCN> create(const std::string &device,
                                           speed_t speed = B115200,
                                           int file_flag = O_RDWR | O_NOCTTY)
    {
        int fd = open(device.c_str(), file_flag);
        if (fd == -1) {
            std::cerr << "[Error] 无法打开串口: " << device << " (errno: " << errno << ")" << std::endl;
            return nullptr;
        }

        auto serial = std::unique_ptr<NT26KCN>(new NT26KCN(fd));
        if (!serial->init(speed)) {
            close(fd);
            return nullptr;
        }

        return serial;
    }

    bool init(speed_t speed) override {
        if (!SerialBase::init(speed)) return false;
        struct termios options;
        if (tcgetattr(serial_fd_, &options) != 0) {
            std::cerr << "[Error] 获取串口属性失败" << std::endl;
            return false;
        }
        // 添加非规范模式的读取设置
        options.c_cc[VMIN] = 2;   // 至少读取2个字符(OK or ERROR)
        options.c_cc[VTIME] = 0;  // 无超时

        if (tcsetattr(serial_fd_, TCSANOW, &options) != 0) {
            std::cerr << "[Error] 设置串口属性失败" << std::endl;
            return false;
        }
        
        tcflush(serial_fd_, TCIOFLUSH);
        return true;
    }
};
