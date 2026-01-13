#ifndef _UART_SERIAL_H
#define _UART_SERIAL_H

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <string>
#include <iostream>

class SerialPort {
private:
    int fd;

public:
    SerialPort(const std::string& device, speed_t speed = B115200) {
        fd = open(device.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd == -1) {
            std::cerr << "[Error] 无法打开串口: " << device << std::endl;
        }

        struct termios options;
        tcgetattr(fd, &options);
        cfsetispeed(&options, speed);
        cfsetospeed(&options, speed);
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_oflag &= ~OPOST;
        tcsetattr(fd, TCSANOW, &options);
    }

    ~SerialPort() {
        if (fd != -1) {
            close(fd);
        }
    }

    bool isValid() const { return fd != -1; }

    void send(const std::string& msg) {
        if (fd != -1) {
            ::write(fd, msg.c_str(), msg.length());
        }
    }
};

#endif