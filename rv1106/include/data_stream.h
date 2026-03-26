#ifndef _DATA_STREAM_H
#define _DATA_STREAM_H

#include <deque>
#include "boost/lockfree/spsc_queue.hpp"
#include "model_process/model_setting.h"
#include "memory.h"

namespace Hip {
    struct DataFrame {
        union {
            #ifndef HIP_MODEL_10_INPUT_CHANNEL
                struct {
                    float acc_x, acc_y, acc_z;
                    float gyro_x, gyro_y, gyro_z;
                    float motorPosL, motorPosR;
                };
            #else
                struct {
                    float acc_x, acc_y, acc_z;
                    float gyro_x, gyro_y, gyro_z;
                    float motorPosL, motorPosR;
                    float motorVelL, motorVelR;
                };
            #endif
            float channels[Hip::ModelChannel];
        };
    };
    inline constexpr int STREAM_LENGTH = 100;
    using DataStream = boost::lockfree::spsc_queue<DataFrame, boost::lockfree::capacity<STREAM_LENGTH>>;
}

namespace Knee {
    struct DataFrame {
    union {
        struct {
            float gyro_x, gyro_y, gyro_z, acc_x;
            float acc_y, acc_z, motorPos, motorVel;
        };
        float channels[8];
    };
};
    inline constexpr int STREAM_LENGTH = 280;
    using DataStream = boost::lockfree::spsc_queue<DataFrame, boost::lockfree::capacity<STREAM_LENGTH>>;
}


struct ImuData {
    union {
        struct {
            uint32_t timestamp;
            float acc_x, acc_y, acc_z;    // m/s^2
            float gyro_x, gyro_y, gyro_z; // dps
        };
        float data[7];
    };
};

struct MotorData {
    union{
        struct {
            float motorPosLeft;     // deg
            float motorVelLeft;     // deg/s 
            float motorPosRight;    // deg
            float motorVelRight;    // deg/s
        };
        float data[4];
    };
};

struct ImuMotorData
{
    struct ImuData imu_data;
    struct MotorData motor_data;
};

struct KneeData {
    union{
        struct {
            float motorPosL;  // 
            float motorVel;  // m/s
            float acc_x;
            float acc_y;
            float acc_z;   // x, y, z
            float gyro_x;   // 
            float gyro_y;
            float gyro_z;  // x, y, z
            float momentRev;
            float time1;
            float time2;
        };
        float data[11];
    };
    
    // uint64_t timestamp_us; // 接收到的系统时间(微秒)
};

#endif