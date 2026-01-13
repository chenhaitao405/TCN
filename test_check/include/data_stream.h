#ifndef _DATA_STREAM_H
#define _DATA_STREAM_H

#include <deque>

struct alignas(16) DataFrame {
    union {
        struct {
            float gyro_x, gyro_y, gyro_z, acc_x;
            float acc_y, acc_z, motorPos, motorVel;
        };
        float channels[8]; // 数组形式，方便 NEON 
    };
};

constexpr int STREAM_LENGTH = 280;
using DataStream = std::deque<DataFrame>;

#endif