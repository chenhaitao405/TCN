#include "gpio/gpio.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

GpioPin::GpioPin(int pin_num, DIRECTION direction, EDGE edge)
    : pin(pin_num), direction(direction), edge(edge) {
    gpio_path = "/sys/class/gpio/gpio" + std::to_string(pin_num);
    value_path = gpio_path + std::string("/value");

    std::ofstream export_file("/sys/class/gpio/export");
    if (export_file.is_open()) {
        export_file << pin;
        export_file.close();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::ofstream dir_file(gpio_path + "/direction");
    if (dir_file.is_open()) {
        switch (direction) {
            case DIRECTION::OUT:
                dir_file << "out";
                break;
            case DIRECTION::IN:
                dir_file << "in";
                break;
        }
        dir_file.close();
    } else {
        std::cerr << "[Error] can't assign GPIO" << pin << " direction, maybe permission denied" << std::endl;
    }

    std::ofstream edge_file(gpio_path + "/edge");
    if (edge_file.is_open()) {
        switch (edge) {
            case EDGE::BOTH:
                edge_file << "both";
                break;
            case EDGE::RISING:
                edge_file << "rising";
                break;
            case EDGE::FALLING:
                edge_file << "falling";
                break;
            case EDGE::None:
                edge_file << "none";
                break;
        }
        edge_file.close();
    } else {
        std::cerr << "[Error] can't assign GPIO" << pin << " edge, maybe permission denied" << std::endl;
    }
}

GpioPin::~GpioPin() {
    std::ofstream unexport_file("/sys/class/gpio/unexport");
    if (unexport_file.is_open()) {
        unexport_file << pin;
    }
}

int GpioPin::read() {
    std::ifstream file(gpio_path + "/value");
    int value = -1;
    if (file.is_open()) {
        file >> value;
    }
    file.close();
    return value;
}

bool GpioPin::write(int value) {
    std::ofstream value_file(gpio_path + "/value");
    if (value_file.is_open()) {
        value_file << (value ? "1" : "0");
        value_file.flush();
    } else {
        return false;
    }
    return true;
}

bool GpioPin::set_io_mode(DIRECTION _direction) {
    std::ofstream dir_file(gpio_path + "/direction");
    if (dir_file.is_open()) {
        switch (_direction) {
            case DIRECTION::IN:
                dir_file << "in";
                break;
            case DIRECTION::OUT:
                dir_file << "out";
                break;
        }
        direction = _direction;
        dir_file.close();
    } else {
        return false;
    }
    return true;
}

bool GpioPin::set_gpio_edge(EDGE _edge) {
    std::ofstream edge_file(gpio_path + "/edge");
    if (edge_file.is_open()) {
        switch (_edge) {
            case EDGE::BOTH:
                edge_file << "both";
                break;
            case EDGE::RISING:
                edge_file << "rising";
                break;
            case EDGE::FALLING:
                edge_file << "falling";
                break;
            case EDGE::None:
                edge_file << "none";
                break;
        }
        edge = _edge;
        edge_file.close();
    } else {
        return false;
    }
    return true;
}

std::string GpioPin::get_value_path() {
    return value_path;
}
