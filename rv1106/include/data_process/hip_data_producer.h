#pragma once
#include <string>
#include "DspFilters/Butterworth.h"
#include "data_stream.h"
#include "peripherals/lsm6dsr.h"
#include "serial/uart_serial.h"

namespace hip_producer {

void sendData(float left_moment, uint8_t pattern, uint8_t fall_back);
bool start_serial2mcu(const std::string& portName);
void stop_serial2mcu();

extern Hip::DataStream data_stream;

} // namespace hip_producer