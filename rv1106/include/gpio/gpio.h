#ifndef _GPIO_H
#define _GPIO_H

#include <string>

class GpioPin
{

public:
    // 触发方式
    enum class EDGE
    {
        RISING,
        FALLING,
        BOTH,
        None
    };
    // IO模式
    enum class DIRECTION
    {
        IN,
        OUT
    };

    GpioPin(int pin_num, DIRECTION direction, EDGE edge = EDGE::None);
    ~GpioPin();

    int read();
    bool write(int value);
    bool set_io_mode(DIRECTION _direction);
    bool set_gpio_edge(EDGE _edge);
    std::string get_value_path();

private:
    int pin;
    std::string gpio_path;
    std::string value_path;
    std::string direction_path;
    DIRECTION direction;
    EDGE edge;
};

#endif