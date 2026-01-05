#ifndef _DATA_STREAM_H
#define _DATA_STREAM_H

#include <deque>

struct DataFrame {
    float gyro_x, gyro_y, gyro_z;
    float acc_x, acc_y, acc_z;
    float motorPos, motorVel;
};

constexpr int STREAM_LENGTH = 280;
using DataStream = std::deque<DataFrame>;

#endif