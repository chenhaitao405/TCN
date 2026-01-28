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
            timeout.tv_usec = 10000;  // 10ms超时，保证能响应running_变化
            
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
                        memcpy(data.data, &frame[2], sizeof(KneeData));

                        // 单位转换
                        data.acc_x = data.acc_x / 1000.0f * 9.81f;
                        data.acc_y = data.acc_y / 1000.0f * 9.81f;
                        data.acc_z = data.acc_z / 1000.0f * 9.81f;
                        data.gyro_x = data.gyro_x / 1000.0f;
                        data.gyro_y = data.gyro_y / 1000.0f;
                        data.gyro_z = data.gyro_z / 1000.0f;
                        
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

class FallDetector {
public:
    FallDetector()
        : g_value_(9.81f),
          weightless_factor_(0.7f),
          impact_factor_(1.5f),
          still_acc_lower_(0.7f * g_value_),
          still_acc_upper_(1.3f * g_value_),
          still_gyro_thresh_(0.35f),
          orientation_diff_thresh_deg_(60.0f),
          stage_(Stage::Idle),
          fall_detected_(false),
          initial_set_(false) {}

    // 每帧调用，返回true表示检测到一次跌倒事件
    bool update(const KneeData& data) {
        using clock = std::chrono::steady_clock;
        const auto now = clock::now();

        const float acc_norm = norm3(&data.acc_x);
        const float gyro_norm = norm3(&data.gyro_x);

        // 采集初始站立姿态
        if (!initial_set_ && acc_norm > 0.5f * g_value_) {
            copy3(&data.acc_x, initial_orientation_);
            normalize3(initial_orientation_);
            initial_set_ = true;
        }

        switch (stage_) {
        case Stage::Idle:
            fall_detected_ = false;
            if (detectWeightless(acc_norm)) {
                stage_ = Stage::Weightless;
                t_weightless_ = now;
            }
            break;

        case Stage::Weightless: {
            const auto dt = now - t_weightless_;
            if (detectImpact(acc_norm) && dt < impact_window_) {
                stage_ = Stage::Impacted;
                t_impact_ = now;
            } else if (dt > max_weightless_span_) {
                reset();
            }
            break;
        }

        case Stage::Impacted: {
            const auto dt = now - t_impact_;
            if (detectStillness(acc_norm, gyro_norm) && dt < still_window_) {
                stage_ = Stage::Still;
                t_still_start_ = now;
            } else if (dt > still_window_) {
                reset();
            }
            break;
        }

        case Stage::Still: {
            const auto still_dt = now - t_still_start_;
            if (detectStillness(acc_norm, gyro_norm)) {
                if (still_dt >= min_still_duration_) {
                    if (orientationChanged(&data.acc_x)) {
                        fall_detected_ = true;
                        reset();
                        return true;
                    } else {
                        reset();
                    }
                }
            } else {
                reset();
            }
            break;
        }
        }
        return false;
    }

    void reset() {
        stage_ = Stage::Idle;
        t_weightless_ = std::chrono::steady_clock::now();
        t_impact_ = t_weightless_;
        t_still_start_ = t_weightless_;
    }

private:
    enum class Stage { Idle, Weightless, Impacted, Still };

    float g_value_;
    float weightless_factor_;
    float impact_factor_;
    float still_acc_lower_;
    float still_acc_upper_;
    float still_gyro_thresh_;
    float orientation_diff_thresh_deg_;

    Stage stage_;
    bool fall_detected_;
    bool initial_set_;
    float initial_orientation_[3] = {0.0f, 0.0f, 0.0f};

    std::chrono::steady_clock::time_point t_weightless_{};
    std::chrono::steady_clock::time_point t_impact_{};
    std::chrono::steady_clock::time_point t_still_start_{};

    // 时间窗口
    const std::chrono::milliseconds impact_window_{800};
    const std::chrono::milliseconds max_weightless_span_{1200};
    const std::chrono::milliseconds still_window_{2500};
    const std::chrono::milliseconds min_still_duration_{600};

    static float norm3(const float v[3]) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }

    static void copy3(const float src[3], float dst[3]) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
    }

    static void normalize3(float v[3]) {
        const float n = norm3(v);
        if (n > 1e-5f) {
            v[0] /= n;
            v[1] /= n;
            v[2] /= n;
        }
    }

    bool detectWeightless(float acc_norm) const {
        return acc_norm < weightless_factor_ * g_value_;
    }

    bool detectImpact(float acc_norm) const {
        return acc_norm > impact_factor_ * g_value_;
    }

    bool detectStillness(float acc_norm, float gyro_norm) const {
        return acc_norm > still_acc_lower_ && acc_norm < still_acc_upper_ && gyro_norm < still_gyro_thresh_;
    }

    bool orientationChanged(const float acc[3]) const {
        if (!initial_set_) return true; // 没有初始参考时，保持敏感
        float curr[3];
        copy3(acc, curr);
        normalize3(curr);
        float dot = initial_orientation_[0] * curr[0] + initial_orientation_[1] * curr[1] + initial_orientation_[2] * curr[2];
        dot = std::max(-1.0f, std::min(1.0f, dot));
        const float angle_deg = std::acos(dot) * 180.0f / 3.14159265f;
        return angle_deg > orientation_diff_thresh_deg_;
    }
};


// ================= 主函数及应用逻辑 =================

bool g_exit_flag = false;
void signalHandler(int signum) {
    g_exit_flag = true;
}

static FallDetector g_fall_detector;
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
    float gyro_x_transformed = data.gyro_y;
    float gyro_y_transformed = -data.gyro_x;
    float gyro_z_transformed = data.gyro_z;
    
    float acc_x_transformed = data.acc_y;
    float acc_y_transformed = -data.acc_x;
    float acc_z_transformed = data.acc_y;
    
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
    g_fall_detector.reset();

    g_driver->setCallback([&](const KneeData& data) {
        static int count = 0;
        if (count++ % 10 == 0) { // 每10帧打印一次
            std::cout << std::fixed << std::setprecision(3)
                      << "Parsed Data -> "
                      << " | motorPosL: " << data.motorPosL
                      << " | motorVel: " << data.motorVel
                      << " | acc_x: " << data.acc_x
                      << " | gyro_x: " << data.gyro_x << std::endl;
        }
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

        if (g_fall_detector.update(data)) {
            printf("[FallDetector] Potential fall detected\n");
        }

        g_prev_data = frame;
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
    g_fall_detector.reset();
}