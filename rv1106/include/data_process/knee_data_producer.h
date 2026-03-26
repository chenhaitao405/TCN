#pragma once
#include <string>
#include "DspFilters/Butterworth.h"
#include "data_stream.h"
#include "peripherals/lsm6dsr.h"
#include "serial/uart_serial.h"

namespace knee_producer {

void sendMoment(float left_moment, float time_val);
bool start_serial2mcu(const std::string& portName);
void stop_serial2mcu();
void set_leg_side(char side);

extern Knee::DataStream data_stream;
} // namespace knee_producer