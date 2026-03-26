#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
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

// ================= 数据结构定义 =================

// 定义IMU的帧头帧尾
#define DATA_HEAD1 0x7C
#define DATA_HEAD2 0xF4
#define DATA_END1  0xE3
#define DATA_END2  0xEC

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
    bool init(const std::string& portName, int baudRate = 921600) {
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

        cfsetospeed(&tty, (speed_t)baudRate);
        cfsetispeed(&tty, (speed_t)baudRate);

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

while (running_) {
            memset(read_buffer_, 0, sizeof(read_buffer_));
            int bytes_read = read(serial_port_, read_buffer_, sizeof(read_buffer_) - 1);

            if (bytes_read <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            if (bytes_read > 200) continue; // 简单过滤异常包

            if (first_in) {
                first_in = false;
                // 首次仅做简单处理或忽略，此处直接压入后续逻辑处理
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

// 模拟的全局CSV文件流
std::ofstream csvFile;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./readDataKnee_L /dev/ttyUSB0" << std::endl;
        return 1;
    }
    
    std::string portName = argv[1];
    
    // 打开CSV文件
    csvFile.open("/home/wlp/Datasets/knee_data_log.csv", std::ios::out | std::ios::trunc);
    
    if (!csvFile.is_open()) {
        std::cerr << "Failed to open CSV file" << std::endl;
        return -1;
    }
    // 写入CSV头
    csvFile << "motorPos,motorVel,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,moment" << std::endl;

    // 实例化驱动
    SerialIMUDriver driver;
    
    // 初始化
    if (!driver.init(portName)) {
        return 1;
    }

    // 设置回调函数：获取并处理返回的数据
    driver.setCallback([&](const KneeData& data) {
        // 1. 打印到终端
        static int count = 0;
        if (count++ % 100 == 0) { // 每100帧打印一次
            std::cout << std::fixed << std::setprecision(3)
                      << "Parsed Data -> "
                      << "M_PosL: " << data.motorPosL 
                      << " | AccX: " << data.acc[0]
                      << " | GyroZ: " << data.gyro[2] 
                      << " | MomentRev: " << data.momentRev << std::endl;
        }

        // 2. 保存到CSV
        if (csvFile.is_open()) {
            csvFile << data.motorPosL << ","
                    << data.motorVel << ","
                    << data.acc[0] << "," << data.acc[1] << "," << data.acc[2] << ","
                    << data.gyro[0] << "," << data.gyro[1] << "," << data.gyro[2] << ","
                    << data.momentRev << std::endl;
        }
    });

    std::cout << "Start reading IMU from " << portName << "..." << std::endl;
    
    // 启动线程
    driver.start();

    // 捕捉 Ctrl+C
    signal(SIGINT, signalHandler);

    // 主线程循环
    while (!g_exit_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // 发送力矩指令
        // driver.sendMoment(5.0f, 0.0f);
    }

    std::cout << "Exiting..." << std::endl;
    driver.stop();
    csvFile.close();

    return 0;
}