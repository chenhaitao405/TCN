#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <csignal>
#include <cmath>
#include "data_stream.h"
#include <memory>
#include "DspFilters/Butterworth.h"

// ================= 数据结构定义 =================

// 定义IMU的帧头帧尾
#define DATA_HEAD1 0x7C
#define DATA_HEAD2 0xF4
#define DATA_END1  0xE3
#define DATA_END2  0xEC

// 存储解析后的数据结构
struct KneeData {
    float motorPosL;
    float motorVel;
    float acc[3];   // x, y, z
    float gyro[3];  // x, y, z
    float momentRev;
    float time1;
    float time2;
    // uint64_t timestamp_us; // 接收到的系统时间(微秒)
};

// 回调函数类型定义
using DataCallback = std::function<void(const KneeData&)>;

// ================= 驱动类定义 =================

class SerialIMUDriver {
public:
    SerialIMUDriver() : running_(false), serial_port_(-1) {}
    
    ~SerialIMUDriver() {
        stop();
    }

    // 初始化串口
    bool init(const std::string& portName, speed_t speed = B921600) {
        serial_port_ = open(portName.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if (serial_port_ < 0) {
            std::cerr << "Error opening serial port: " << portName << std::endl;
            return false;
        }

        struct termios tty;
        memset(&tty, 0, sizeof(tty));
        if (tcgetattr(serial_port_, &tty) != 0) {
            std::cerr << "Error getting serial attributes" << std::endl;
            return false;
        }

        cfsetospeed(&tty, speed);
        cfsetispeed(&tty, speed);

        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
        tty.c_cflag |= (CLOCAL | CREAD);                // ignore modem controls, enable reading
        tty.c_cflag &= ~(PARENB | PARODD);              // no parity
        tty.c_cflag &= ~CSTOPB;                         // 1 stop bit
        tty.c_cflag &= ~CRTSCTS;                        // no flow control

        // raw mode
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
        tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
        tty.c_oflag &= ~OPOST;

        // blocking read settings
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 1; // 0.1s timeout

        if (tcsetattr(serial_port_, TCSANOW, &tty) != 0) {
            std::cerr << "Error setting serial attributes" << std::endl;
            return false;
        }
        
        // 刷新缓冲区
        tcflush(serial_port_, TCIOFLUSH);
        return true;
    }

    // 注册数据回调函数
    void setCallback(DataCallback cb) {
        callback_ = cb;
    }

    // 启动接收线程
    void start() {
        if (running_) return;
        running_ = true;
        worker_thread_ = std::thread(&SerialIMUDriver::processLoop, this);
    }

    // 停止接收
    void stop() {
        if (!running_) return;
        running_ = false;
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        if (serial_port_ >= 0) {
            close(serial_port_);
            serial_port_ = -1;
        }
    }

    // 发送力矩指令 (替代原ROS订阅后的操作)
    void sendMoment(float left_moment, float time_val) {
        if (serial_port_ < 0) return;

        left_moment = std::max(-18.0f, std::min(18.0f, left_moment));
        int8_t lef_m = static_cast<int8_t>(left_moment / 18.0f * 127);
        
        uint8_t buffer_send[6] = {0xA1, 0x00, 0x00, 0x00, 0x00, 0x0a}; // 示例协议
        buffer_send[1] = (uint8_t)lef_m;
        memcpy(&buffer_send[2], &time_val, 4);
        write(serial_port_, buffer_send, 6);
    }

private:
    int serial_port_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
    DataCallback callback_;
    std::deque<uint8_t> buff_dy_;
    uint8_t read_buffer_[256];

    // 大端序转换
    uint32_t swapU32(uint32_t val) {
        return ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) | 
               ((val << 8) & 0xFF0000) | ((val << 24) & 0xFF000000);
    }

    void processLoop() {
        bool first_in = true;
        bool find_new_head = false;
        
        fd_set read_fds;
        struct timeval timeout;

        while (running_) {
            // ========== 事件驱动：等待串口可读 ==========
            FD_ZERO(&read_fds);
            FD_SET(serial_port_, &read_fds);
            timeout.tv_sec = 0;
            timeout.tv_usec = 10000;  // 10ms 超时，保证能响应 running_ 变化
            
            int ready = select(serial_port_ + 1, &read_fds, nullptr, nullptr, &timeout);
            
            if (ready <= 0) {
                // 超时或错误，直接继续（无需额外 sleep）
                continue;
            }
            
            // 背压控制：缓冲区积压过多时短暂让出CPU
            if (buff_dy_.size() > 384) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            memset(read_buffer_, 0, sizeof(read_buffer_));
            int bytes_read = read(serial_port_, read_buffer_, sizeof(read_buffer_) - 1);

            if (bytes_read <= 0) continue;  // select 说有数据但读不到，跳过
            if (bytes_read > 200) continue; // 简单过滤异常包

            if (first_in) {
                first_in = false;
            }

            for (int i = 0; i < bytes_read; ++i) {
                buff_dy_.push_back(read_buffer_[i]);
            }

            while (buff_dy_.size() >= 48) {
                if (find_new_head) {
                    find_new_head = false;
                    while (!buff_dy_.empty()) {
                        if (buff_dy_.front() == DATA_HEAD1) {
                            if (buff_dy_.size() > 1) {
                                auto it = buff_dy_.begin();
                                if (*(++it) == DATA_HEAD2) break; 
                            } else break; 
                        }
                        buff_dy_.pop_front();
                    }
                    if (buff_dy_.size() < 48) break;
                }

                // 预览帧数据 (先不pop，确认完整再pop)
                uint8_t frame[48];
                for (int i = 0; i < 48; ++i) frame[i] = buff_dy_[i];

                if (frame[0] == DATA_HEAD1 && frame[1] == DATA_HEAD2) {
                    if (frame[46] == DATA_END1 && frame[47] == DATA_END2) {
                        // 校验成功，从队列移除
                        for(int i=0; i<48; i++) buff_dy_.pop_front();

                        KneeData data;
                        memcpy(&data.motorPosL, &frame[2], 4);
                        memcpy(&data.motorVel, &frame[6], 4);
                        
                        memcpy(&data.acc[0], &frame[10], 4); 
                        memcpy(&data.acc[1], &frame[14], 4); 
                        memcpy(&data.acc[2], &frame[18], 4); 
                        
                        memcpy(&data.gyro[0], &frame[22], 4); 
                        memcpy(&data.gyro[1], &frame[26], 4); 
                        memcpy(&data.gyro[2], &frame[30], 4); 
                        
                        memcpy(&data.momentRev, &frame[34], 4);
                        memcpy(&data.time1, &frame[38], 4);
                        memcpy(&data.time2, &frame[42], 4);

                        // 单位转换
                        for(int i=0; i<3; i++) data.acc[i] = data.acc[i] / 1000.0f * 9.81f;
                        for(int i=0; i<3; i++) data.gyro[i] = data.gyro[i] / 1000.0f;

                        if (callback_) callback_(data);

                    } else {
                        // 帧尾错误，丢弃头部，寻找下一个头
                        buff_dy_.pop_front();
                        find_new_head = true;
                    }
                } else {
                    // 帧头错误
                    buff_dy_.pop_front();
                    find_new_head = true;
                }
            }
        }
    }
};

// ================= 主函数及应用逻辑 =================

bool g_exit_flag = false;
void signalHandler(int signum) {
    g_exit_flag = true;
}

static DataStream g_data_stream;
static std::mutex g_data_stream_mutex;
static std::unique_ptr<SerialIMUDriver> g_driver;

// Butterworth 低通滤波器 (2阶)
static Dsp::SimpleFilter<Dsp::Butterworth::LowPass<2>, 1> g_motorvel_filter;
static bool g_filter_initialized = false;

// 用于线性插值的上一帧数据
static DataFrame g_prev_data;
static bool g_has_prev_data = false;

// 侧边标识：'l' 左腿, 'r' 右腿
static char g_imu_side = 'l';

// 设置侧边标识
void set_imu_side(char side) {
    g_imu_side = side;
}

static void interpolate_frame(const DataFrame& prev, const DataFrame& curr, float t, DataFrame& out) {
    out.gyro_x = prev.gyro_x + t * (curr.gyro_x - prev.gyro_x);
    out.gyro_y = prev.gyro_y + t * (curr.gyro_y - prev.gyro_y);
    out.gyro_z = prev.gyro_z + t * (curr.gyro_z - prev.gyro_z);
    out.acc_x = prev.acc_x + t * (curr.acc_x - prev.acc_x);
    out.acc_y = prev.acc_y + t * (curr.acc_y - prev.acc_y);
    out.acc_z = prev.acc_z + t * (curr.acc_z - prev.acc_z);
    out.motorPos = prev.motorPos + t * (curr.motorPos - prev.motorPos);
    out.motorVel = prev.motorVel + t * (curr.motorVel - prev.motorVel);
}

inline void transform_frame(const KneeData& data, DataFrame& transformed_data) {
    // 1. 坐标系旋转变换
    // paper_x = device_y, paper_y = -device_x, paper_z = device_z
    float gyro_x_transformed = data.gyro[1];
    float gyro_y_transformed = -data.gyro[0];
    float gyro_z_transformed = data.gyro[2];
    
    float acc_x_transformed = data.acc[1];
    float acc_y_transformed = -data.acc[0];
    float acc_z_transformed = data.acc[2];
    
    // 2. 左腿镜像处理
    if (g_imu_side == 'l') {
        gyro_x_transformed *= -1.0f;
        gyro_y_transformed *= -1.0f;
        acc_z_transformed *= -1.0f;
    }
    
    transformed_data.gyro_x = gyro_x_transformed;
    transformed_data.gyro_y = gyro_y_transformed;
    transformed_data.gyro_z = gyro_z_transformed;
    transformed_data.acc_x = acc_x_transformed;
    transformed_data.acc_y = acc_y_transformed;
    transformed_data.acc_z = acc_z_transformed;
    
    // 3. 电机角度减180度
    transformed_data.motorPos = data.motorPosL - 180.0f;
    
    // 4. 电机速度除以2
    transformed_data.motorVel = data.motorVel / 2.0f;
}

// 对外暴露数据流与互斥量
DataStream& get_data_stream() { return g_data_stream; }
std::mutex& get_data_stream_mutex() { return g_data_stream_mutex; }

// 发送力矩指令接口
void send_moment(float moment, float time_val) {
    if (g_driver) {
        g_driver->sendMoment(moment, time_val);
    }
}

// 启停驱动接口
bool start_imu_driver(const std::string& portName) {
    g_driver = std::make_unique<SerialIMUDriver>();
    if (!g_driver->init(portName)) {
        g_driver.reset();
        return false;
    }
    
    // 初始化滤波器：2阶，采样率200Hz（重采样后），截止频率10Hz
    if (!g_filter_initialized) {
        g_motorvel_filter.setup(2, 200.0, 10.0);
        g_filter_initialized = true;
    }
    
    g_has_prev_data = false;
    
    g_driver->setCallback([&](const KneeData& data) {
        DataFrame interp_frame, frame;
        transform_frame(data, frame);
        
        // 对插值帧的motorVel进行滤波
        if (g_has_prev_data) {
            interpolate_frame(g_prev_data, frame, 0.5f, interp_frame);
            float* ch_interp[1] = {&interp_frame.motorVel};
            g_motorvel_filter.process(1, ch_interp);
        }
        
        // 对当前帧motorVel进行滤波
        float* ch_curr[1] = {&frame.motorVel};
        g_motorvel_filter.process(1, ch_curr);

        {
            std::lock_guard<std::mutex> lk(g_data_stream_mutex);
            if (g_has_prev_data) {
                g_data_stream.push_back(interp_frame);
                if (g_data_stream.size() > STREAM_LENGTH) {
                    g_data_stream.pop_front();
                }
            }

            g_data_stream.push_back(frame);
            if (g_data_stream.size() > STREAM_LENGTH) {
                g_data_stream.pop_front();
            }
        }

        g_prev_data = std::move(frame);
        g_has_prev_data = true;
    });

    g_driver->start();
    return true;
}

void stop_imu_driver() {
    if (g_driver) {
        g_driver->stop();
        g_driver.reset();
    }
    // 重置滤波器状态和插值状态
    g_filter_initialized = false;
    g_has_prev_data = false;
}