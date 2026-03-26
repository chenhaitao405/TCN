#pragma once

namespace Hip {
    #ifndef HIP_MODEL_10_INPUT_CHANNEL
        inline constexpr int ModelChannel = 8;
        inline constexpr float g_mean_vals[ModelChannel] = {
            0.013860977254807949,
            0.3187899589538574,
            -9.661064147949219,
            0.9842904210090637,
            -0.659106969833374,
            -0.6765093803405762,
            0.20177024602890015,
            0.22021803259849548
        };
        inline constexpr float g_std_vals[ModelChannel] = {
            1.6076325178146362,
            1.6515476703643799,
            2.277411460876465,
            14.99991226196289,
            17.85957145690918,
            27.48878288269043,
            0.24502652883529663,
            0.253909170627594
        };
    #else 
        inline constexpr int ModelChannel = 10;
        inline constexpr float g_mean_vals[ModelChannel] = {
            0.013860977254807949,
            0.3187899589538574,
            -9.661064147949219,
            0.9842904210090637,
            -0.659106969833374,
            -0.6765093803405762,
            0.20177024602890015,
            0.22021803259849548,
            0,
            0
        };
        inline constexpr float g_std_vals[ModelChannel] = {
            1.6076325178146362,
            1.6515476703643799,
            2.277411460876465,
            14.99991226196289,
            17.85957145690918,
            27.48878288269043,
            0.24502652883529663,
            0.253909170627594,
            1,
            1
        };
    #endif
    
    inline constexpr int NumClasses = 4;
    
    inline constexpr const char* ACTION_NAMES[Hip::NumClasses] = {
    "upstair", "downstair", "walk", "stand"
};
}

namespace Knee {
    inline constexpr float g_mean_vals[8] = {
        0.7113f, -0.4300f, 0.7254f, // gyro
        2.6529f, 8.7143f, -0.2818f, // acc
        -30.6663f, -0.1348f         // motorpos motorvel
    };
    inline constexpr const float g_std_vals[8] = {
        21.3148f, 45.2281f, 81.4140f, // gyro
        3.9808f, 4.4334f, 1.8336f,    // acc
        27.8284f, 107.1712f           // // motorpos motorvel
    };

    inline constexpr int NumClasses = 4;
    inline constexpr int ModelChannel = 8;
}