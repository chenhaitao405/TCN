#pragma once

// #########################################################################################
//                                  Register Mapping
// #########################################################################################

/*
FUNC_CFG_ACCESS | SHUB_REG_ACCESS | 0 | 0 | 0 | 0 | 0 | 0

FUNC_CFG_ACCESS 位控制嵌入式功能配置寄存器访问 (default : 0)
    该位为 0 时，禁止访问嵌入式功能相关配置寄存器（如计步器、有限状态机 FSM、显著运动检测等功能的配置寄存器）。
    该位为 1 时，允许读写嵌入式功能配置寄存器。
SHUB_REG_ACCESS 位控制传感器中枢寄存器访问 (default : 0)
    该位为 0 时，禁止访问传感器中枢（I2C 主机）相关寄存器。
    该位为 1 时，允许读写传感器中枢寄存器，支持配置外部传感器连接、I2C 主机通信等功能。
*/
#define FUNC_CFG_ACCESS 0x01

/*
OIS_PU_DIS | SDO_PU_EN | 1 | 1 | 1| 1 | 1 | 1

OIS_PU_DIS 位控制OCS_Aux 与 SDO_Aux 引脚的上拉禁用 (default : 0)
    该位为 0 时，OCS_Aux 和 SDO_Aux 引脚启用上拉电阻。
    该位为 1 时，OCS_Aux 和 SDO_Aux 引脚的上拉电阻断开，适用于不需要上拉的应用场景。
SDO_PU_EN 位控制SDO 引脚的上拉使能 (default : 0)
    该位为 0 时，SDO 引脚的上拉电阻断开。
    该位为 1 时，SDO 引脚启用上拉电阻，可增强该引脚的信号稳定性，避免电平漂移。
*/
#define PIN_CTRL 0x02

/*
TPH_H_SEL | TPH_L_6 | TPH_L_5 | TPH_L_4 | TPH_L_3 | TPH_L_2 | TPH_L_1 | TPH_L_0

TPH_H_SEL 位控制是否启用 TPH_H 寄存器参与时间帧计算。
    该位为 1 时，时间帧样本数由公式 TPH [#Samples] = 2 x (TPH_L + 256 x TPH_H) 计算。
    该位为 0 时，时间帧样本数仅由 S4S_TPH_L 决定，公式简化为 TPH [#Samples] = 2 x TPHL
TPH_L [6:0] 位配置基础时间帧
    这 7 位用于设置 S4S 同步的基础样本数（d0），是时间帧计算的核心参数。
    若 TPH_H_SEL=0 且 TPH_L [6:0] = 0，则 S4S 功能被禁用。
*/
#define S4S_TPH_L 0x04

/*
TPH_H_7 | TPH_H_6 | TPH_H_5 | TPH_H_4 | TPH_H_3 | TPH_H_2 | TPH_H_1 | TPH_H_0

TPH_H_[7:0] 为 8 位高字节寄存器，与 S4S_TPH_L（低字节）共同组成 16 位时间帧参数。
*/
#define S4S_TPH_H 0x05

/*
0 | 0 | 0 | 0 | 0 | 0 | RR_1 | RR_0

RR_[1:0] 用于定义 S4S 同步的 DT 分辨率，不同组合对应不同精度：
    00: S4S, DT resolution 2^11 ;
    01: S4S, DT resolution 2^12 ;
    10: S4S, DT resolution 2^13 ;
    11: S4S, DT resolution 2^14
*/
#define S4S_RR 0x06

/*
WTM7 | WTM6 | WTM5 | WTM4 | WTM3 | WTM2 | WTM1 | WTM0

WTM[7:0] 控制FIFO水位线阈值的低8位，与 FIFO_CTRL2 中的 WTM8 位组成完整的9位阈值参数。
    1 LSB = 1 sensor (6 bytes) + TAG (1 byte) written in FIFO
    当 FIFO 中未读数据量达到或超过该阈值时，会触发 FIFO 水印中断
*/
#define FIFO_CTRL1 0x07

/*
STOP_ON_WTM | FIFO_COMPR_RT_EN | 0 | ODRCHG_EN | 0 | UNCOPTR_RATE_1 | UNCOPTR_RATE_0 | WTM8

STOP_ON_WTM 控制存储停止 (default : 0)
    该位为 0 时，FIFO 存储不受阈值限制，直至填满。
    该位为 1 时，FIFO 存储到水印阈值后停止，避免数据溢出，需手动读取后重新启动。
FIFO_COMPR_RT_EN 控制启用 FIFO 运行时压缩功能。
    该位为 0 时，不启用压缩功能。
    该位为 1 时，启用压缩功能，

*/
#define FIFO_CTRL2 0x08

/*
BDR_GY_3 | BDR_GY_2 | BDR_GY_1 | BDR_GY_0 | BDR_XL_3 | BDR_XL_2 | BDR_XL_1 | BDR_XL_0

BDR_GY_[3:0]：陀螺仪批量数据率（FIFO 写入频率）选择
    用于配置陀螺仪数据存入 FIFO 的频率，默认值 0000（不存入 FIFO）
        0000：陀螺仪不存入 FIFO（默认）
        0001：12.5 Hz
        0010：26 Hz
        0011：52 Hz
        0100：104 Hz
        0101：208 Hz
        0110：417 Hz
        0111：833 Hz
        1000：1667 Hz
        1001：3333 Hz
        1010：6667 Hz
        1011：6.5 Hz
        1100-1111：不允许配置
BDR_XL_[3:0]：加速度计批量数据率（FIFO 写入频率）选择
    用于配置加速度计数据存入 FIFO 的频率，默认值 0000（不存入 FIFO）
        0000：加速度计不存入 FIFO（默认）
        0001：12.5 Hz
        0010：26 Hz
        0011：52 Hz
        0100：104 Hz
        0101：208 Hz
        0110：417 Hz
        0111：833 Hz
        1000：1667 Hz
        1001：3333 Hz
        1010：6667 Hz
        1011：1.6 Hz
        1100-1111：不允许配置

*/
#define FIFO_CTRL3 0x09

/*
DEC_TS_BATCH_1 | DEC_TS_BATCH_0 | ODR_T_BATCH_1 | ODR_T_BATCH_0 | 0 | FIFO_MODE_2 | FIFO_MODE_1 | FIFO_MODE_0

DEC_TS_BATCH_[1:0]：时间戳批量抽取系数选择。
    用于配置时间戳存入 FIFO 的抽取比例，默认值 00（不存入 FIFO）。
        00：时间戳不存入 FIFO（默认）
        01：抽取系数 1（存入频率 = 加速度计与陀螺仪批量率的最大值）
        10：抽取系数 8（存入频率 = 最大值 / 8）
        11：抽取系数 32（存入频率 = 最大值 / 32）
ODR_T_BATCH_[1:0]：温度数据批量率（FIFO 写入频率）选择。
    用于配置温度数据存入 FIFO 的频率，默认值 00（不存入 FIFO）。
        00：温度数据不存入 FIFO（默认）
        01：1.6 Hz
        10：12.5 Hz
        11：52 Hz
FIFO_MODE_[2:0]：FIFO 工作模式选择
    用于配置 FIFO 的运行模式，默认值 000（旁路模式）。
        000：旁路模式（FIFO 禁用，始终为空）
        001：FIFO 模式（存满后停止存储）
        010：Reserved;
        011：连续到 FIFO 模式（触发信号无效时连续模式，有效时 FIFO 模式）
        100：旁路到连续模式（触发信号无效时旁路模式，有效时连续模式）
        101：Reserved;
        110：连续模式（存满后覆盖 oldest 数据）
        111：旁路到 FIFO 模式（触发信号无效时旁路模式，有效时 FIFO 模式）
*/
#define FIFO_CTRL4 0x0A

#define COUNTER_BDR_REG1 0X0B

#define COUNTER_BDR_REG2 0x0C

/*
DEN_DRDY_flag | INT1_CNT_BDR | INT1_FIFO_FULL | INT1_FIFO_OVR | INT1_FIFO_TH | INT1_BOOT | INT1_DRDY_G | INT1_DRDY_XL

DEN_DRDY_flag：DEN 数据就绪标志输出控制。 default : 0
    控制是否将 DEN_DRDY（传感器数据上的 DEN 标记就绪）信号输出到 INT1 引脚。
        该位为 1 时，DEN 标记的数据就绪后会触发 INT1 引脚；
        该位为 0 时，不触发。适用于需要同步 DEN 信号与传感器数据的场景（如 OIS 功能中的数据同步）。
INT1_CNT_BDR：计数器 BDR 中断路由控制。 default : 0
    控制是否将 COUNTER_BDR_IA 中断（批量数据率计数器达到阈值）路由到 INT1 引脚。
        该位为 1 时，当计数器达到 CNT_BDR_TH 阈值，INT1 引脚触发中断；
        该位为 0 时，中断不路由。用于批量数据采集时的定时触发读取。
INT1_FIFO_FULL：FIFO 满中断路由控制。 default : 0
    控制是否将 FIFO_FULL_IA 中断（FIFO 存储满）路由到 INT1 引脚，支持 MIPI I3C^SM 接口的带内中断（IBI）触发。
        该位为 1 时，FIFO 存满后 INT1 引脚触发中断（或 IBI）；
        该位为 0 时，中断不路由。避免 FIFO 数据溢出，适用于大容量数据缓存场景。
INT1_FIFO_OVR：FIFO 溢出中断路由控制。 default : 0
    控制是否将 FIFO_OVR_IA 中断（FIFO 数据溢出）路由到 INT1 引脚，支持 MIPI I3C^SM 接口的带内中断（IBI）触发。
        该位为 1 时，FIFO 数据溢出后 INT1 引脚触发中断（或 IBI）；
        该位为 0 时，中断不路由。用于检测数据丢失，保障数据完整性。
INT1_FIFO_TH：FIFO 阈值中断路由控制。 default : 0
    控制是否将 FIFO_WTM_IA 中断（FIFO 达到水印阈值）路由到 INT1 引脚，支持 MIPI I3C^SM 接口的带内中断（IBI）触发。
        该位为 1 时，FIFO 未读数据量达到阈值后 INT1 引脚触发中断（或 IBI）；
        该位为 0 时，中断不路由。
INT1_BOOT：启动状态中断路由控制。 default : 0
    控制是否将启动状态信号路由到 INT1 引脚。
        该位为 1 时，器件启动完成后 INT1 引脚触发中断；
        该位为 0 时，不触发。用于系统确认器件是否就绪。
INT1_DRDY_G：陀螺仪数据就绪中断路由控制
    控制是否将陀螺仪数据就绪中断路由到 INT1 引脚，支持 MIPI I3C^SM 接口的带内中断（IBI）触发。
        该位为 1 时，陀螺仪产生新数据后 INT1 引脚触发中断（或 IBI）；
        该位为 0 时，中断不路由。适用于需要实时获取陀螺仪数据的场景（如运动控制）。
INT1_DRDY_XL：加速度计数据就绪中断路由控制
    控制是否将加速度计数据就绪中断路由到 INT1 引脚，支持 MIPI I3C^SM 接口的带内中断（IBI）触发。
        该位为 1 时，加速度计产生新数据后 INT1 引脚触发中断（或 IBI）；
        该位为 0 时，中断不路由。适用于需要实时获取加速度数据的场景（如姿态检测）。

*/
#define INT1_CTRL 0x0D

#define INT2_CTRL 0x0E

#define WHO_AM_I 0x0F
/*
ODR_XL3 | ODR_XL2 | ODR_XL1 | ODR_XL0 | FS1_XL | FS0_XL | LPF2_XL_EN | 0

ODR_XL[3:0] Accelerometer ODR selection. See under "Accelerometer ODR" Section
FS[1:0]_XL Accelerometer full-scale selection. Default value: 00.
    00: ±2 g;
    01: ±16 g;
    10: ±4 g;
    11: ±8 g
LPF2_XL_EN Accelerometer high-resolution selection
    0: output from first stage digital filtering selected (default);
    1: output from LPF2 second filtering stage selected
*/
#define CTRL1_XL 0x10

/*
ODR_G3 | ODR_G2 | ODR_G1 | ODR_G0 | FS1_G | FS0_G | FS_125 | FS_4000

ODR_G[3:0] Gyroscope output data rate selection. See under "Gyroscope ODR" Section
FS[1:0]_G Gyroscope UI chain full-scale selection
    00: 250 dps;
    01: 500 dps;
    10: 1000 dps;
    11: 2000 dps
FS_125 Selects gyro UI chain full-scale 125 dps
    0: FS selected through bits FS[1:0]_G;
    1: FS set to 125 dps
FS_4000 Selects gyro UI chain full-scale 4000 dps
    0: FS selected through bits FS[1:0]_G or FS_125;
    1: FS set to 4000 dps
*/
#define CTRL2_G 0x11

/*
    BOOT | BDU | H_LACTIVE | PP_OD | SIM | IF_INC | 0 | SW_RESET
    Bit 6 (BDU) Block Data Update.
        0: continuous update;
        1: output registers are not updated until MSB and LSB have been read
    Bit 5 (H_LACTIVE) = 0 Interrupt activation level.
        0: interrupt output pins active high;
        1: interrupt output pins active low
    Bit 4 (PP_OD) Push-pull/open-drain selection on INT1 and INT2 pins.
        0: push-pull mode;
        1: open-drain mode
    Bit 0 (SW_RESET) Software reset.
        0: normal mode;
         1: reset device
*/
#define CTRL3_C 0x12

/*

*/
#define CTRL4_C 0x13

#define CTRL5_C 0x14

#define CTRL6_C 0x15

#define CTRL7_G 0x16

#define CTRL8_XL 0x17

#define CTRL9_XL 0x18

#define CTRL10_C 0x19

#define ALL_INT_SRC 0x1A

#define WAKE_UP_SRC 0x1B

#define TAP_SRC 0x1C

#define D6D_SRC 0x1D

/*

*/
#define STATUS_REG 0x1E

#define STATUS_SPIAux 0x1E

#define OUT_TEMP_L 0x20

#define OUT_TEMP_H 0x21

#define OUTX_L_G 0x22

#define OUTX_H_G 0x23

#define OUTY_L_G 0x24

#define OUTY_H_G 0x25

#define OUTZ_L_G 0x26

#define OUTZ_H_G 0x27

#define OUTX_L_A 0x28

#define OUTX_H_A 0x29

#define OUTY_L_A 0x2A

#define OUTY_H_A 0x2B

#define OUTZ_L_A 0x2C

#define OUTZ_H_A 0x2D

#define EMB_FUNC_STATUS_MAINPAGE 0x35

#define FSM_STATUS_A_MAINPAGE 0x36

#define FSM_STATUS_B_MAINPAGE 0x37

#define STATUS_MASTER_MAINPAGE 0x39

/*

*/
#define FIFO_STATUS1 0x3A

/*

*/
#define FIFO_STATUS2 0x3B

#define TIMESTAMP0 0x40

#define TIMESTAMP1 0x41

#define TIMESTAMP2 0x42

#define TIMESTAMP3 0x43

#define TAP_CFG0 0x56

#define TAP_CFG1 0x57

#define TAP_CFG2 0x58

#define TAP_THS_6D 0x59

#define INT_DUR2 0x5A

#define WAKE_UP_THS 0x5B

#define WAKE_UP_DUR 0x5C

#define FREE_FALL 0X5D

#define MD1_CFG 0x5E

#define MD2_CFG 0x5F

#define S4S_ST_CMD_CODE 0x60

#define S4S_DT_REG 0x61

#define I3C_BUS_AVB 0x62

#define INTERNAL_FREQ_FINE 0x63

#define INT_OIS 0x6F

#define CTRL1_OIS 0x70

#define CTRL2_OIS 0x71

#define CTRL3_OIS 0x72

#define X_OFS_USR 0x73

#define Y_OFS_USR 0x74

#define Z_OFS_USR 0x75

/*

*/
#define FIFO_DATA_OUT_TAG 0x78

#define FIFO_DATA_OUT_X_L 0x79

#define FIFO_DATA_OUT_X_H 0x7A

#define FIFO_DATA_OUT_Y_L 0x7B

#define FIFO_DATA_OUT_Y_H 0x7C

#define FIFO_DATA_OUT_Z_L 0x7D

#define FIFO_DATA_OUT_Z_H 0x7E



// #########################################################################################
//                                  Sensor Sensitivity
// #########################################################################################

// mg/LSB
#define SENSITIVITY_XL_2G 0.061f
#define SENSITIVITY_XL_4G 0.122f
#define SENSITIVITY_XL_8G 0.244f
#define SENSITIVITY_XL_16G 0.488f

// mdps/LSB
#define SENSITIVITY_G_125   4.375f
#define SENSITIVITY_G_250   8.75f
#define SENSITIVITY_G_500   17.50f
#define SENSITIVITY_G_1000  35.00f
#define SENSITIVITY_G_2000  70.00f
#define SENSITIVITY_G_4000  140.00f


// #########################################################################################
//                                 Accelerometer ODR
// #########################################################################################
#define ACC_POWER_DOWN 0x00
#define ACC_ODR_1_6 0xB0
#define ACC_ODR_12_5 0x10
#define ACC_ODR_26 0x20
#define ACC_ODR_52 0x30
#define ACC_ODR_104 0x40
#define ACC_ODR_208 0x50
#define ACC_ODR_416 0x60
#define ACC_ODR_833 0x70
#define ACC_ODR_1660 0x80
#define ACC_ODR_3330 0x90
#define ACC_ODR_6660 0xA0

// #########################################################################################
//                                 Accelerometer Scale
// #########################################################################################
#define ACC_2g 0x00
#define ACC_4g 0x08
#define ACC_8g 0x0C
#define ACC_16g 0x04

// #########################################################################################
//                                  Gyroscope ODR
// #########################################################################################
#define GYRO_POWER_DOWN 0x00
#define GYRO_ODR_12_5 0x10
#define GYRO_ODR_26 0x20
#define GYRO_ODR_52 0x30
#define GYRO_ODR_104 0x40
#define GYRO_ODR_208 0x50
#define GYRO_ODR_416 0x60
#define GYRO_ODR_833 0x70
#define GYRO_ODR_1660 0x80
#define GYRO_ODR_3330 0x90
#define GYRO_ODR_6660 0xA0

// #########################################################################################
//                                 Gyroscope Scale
// #########################################################################################
#define GYRO_125dps 0x02
#define GYRO_250dps 0x00
#define GYRO_500dps 0x04
#define GYRO_1000dps 0x08
#define GYRO_2000dps 0x0C
#define GYRO_4000dps 0x01

#include <stdint.h>
#include "i2c_bus.h"
#include <vector>
struct ImuData {
    float ax, ay, az; // m/s^2
    float gx, gy, gz; // dps
    uint32_t timestamp;
};

class Lsm6dsr: public I2cDevice{
private:
    bool _debug;
public:
    explicit Lsm6dsr(int bus, uint8_t addr, bool debug=false);

    bool init(uint8_t AccFlag, uint8_t GyroFlag, uint8_t FifoRate);

    bool checkId();

    bool reset();

    bool is_debug() {return _debug;}
    void clearInterrupt();

    std::vector<ImuData> readFifo();
    
    void printFifoStatus();
    
    void startAccelerometer(uint8_t flag);

    void startGyroscope(uint8_t flag);

    void stopAndClearFifo();

    void stopAccelerometer();

    void stopGyroscope();
    
    void startFifo();
};