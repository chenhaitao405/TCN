#ifndef _GPIO_H
#define _GPIO_H

#include <string>
#include <fstream>
#include <thread>
#include <chrono>
#include <iostream>

class GpioPin {
private:
    int pin;
    std::string gpio_path;
    bool out_mode;

public:
    GpioPin(int pin_num, bool out_mode) : pin(pin_num), out_mode(out_mode) {
        gpio_path = "/sys/class/gpio/gpio" + std::to_string(pin_num);
        std::ofstream export_file("/sys/class/gpio/export");
        if (export_file.is_open()) {
            export_file << pin;
            export_file.close();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::ofstream dir_file(gpio_path + "/direction");
        if (dir_file.is_open()) {
            if (out_mode) dir_file << "out";
            else dir_file << "in";
            dir_file.close();
        } else {
            std::cerr << "[Error] can't assign GPIO" << pin << " direction, maybe permission denied" << std::endl;
        }
    }

    ~GpioPin() {
        std::ofstream unexport_file("/sys/class/gpio/unexport");
        if (unexport_file.is_open()) {
            unexport_file << pin;
        }
    }

    bool write(int value) {
        std::ofstream value_file(gpio_path + "/value");
        if (value_file.is_open()) {
            value_file << (value ? "1" : "0");
            value_file.flush();
        } else return false;
        return true;
    }

    bool set_io_mode(bool out_mode) {
        std::ofstream dir_file(gpio_path + "/direction");
        if (dir_file.is_open()) {
            if (out_mode) dir_file << "out";
            else dir_file << "in";
            dir_file.close();
        } else return false;
        return true;
    }
};

#endif