#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <string.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <queue>
#include <deque>
#include <csignal>
#include <thread>
#include <mutex>
#include <vector>
#include <cmath>
#include <condition_variable>
#include <ncurses.h>

// 全局变量控制退出
bool flg_exit = false;
std::condition_variable sig_buffer;
bool keyBoardCatch = true; 

// 用于在ncurses界面显示信息的函数
void display_message(const std::string& msg) {
    if(keyBoardCatch) {
        addstr(msg.c_str());
        addch('\n');
        refresh();
    } else {
        std::cout << msg << std::endl;
    }
}

// 串口相关定义
#define IMU1_HEAD1 0x7f
#define IMU1_HEAD2 0xf1
#define IMU1_END1  0xe0
#define IMU1_END2  0xef

#define IMU2_HEAD1 0x7e
#define IMU2_HEAD2 0xf2
#define IMU2_END1  0xe1
#define IMU2_END2  0xee

#define IMU3_HEAD1 0x7d
#define IMU3_HEAD2 0xf3
#define IMU3_END1  0xe2
#define IMU3_END2  0xed

#define IMU4_HEAD1 0x7c
#define IMU4_HEAD2 0xf4
#define IMU4_END1  0xe3
#define IMU4_END2  0xec

#define MOTOR_HEAD1 0x6f
#define MOTOR_HEAD2 0xe1
#define MOTOR_END1  0xf0
#define MOTOR_END2  0xfe

#define RESULT_HEAD 0xa1
#define RESULT_END  0xaf

// 定义IMU数据结构
struct IMUData {
    float roll, pitch, yaw;
    float acc_x, acc_y, acc_z;
    float gyro_x, gyro_y, gyro_z;
};

// 全局数据存储 (替代 ros 的 wgg_msg)
// 0-1: Motor Angle (L, R)
// 2-3: Motor Vel (L, R)
// 4-5: Unused
// 6-8: IMU1 Acc
// 9-11: IMU1 Gyro
// 12-14: IMU2 Acc
// 15-17: IMU2 Gyro
// 18-19: Motor Torque (L, R)
std::vector<float> global_data_buffer(20, 0.0f);
std::mutex data_mutex; // 线程锁，防止读写冲突

// 全局配置
int flag_num = 0;
bool imu1_debug_out = 0;
bool motor_debug_out = 0;
int serial_port;

// 文件流
std::fstream imuEulerStream1, imuEulerStream2, imuEulerStream3, imuEulerStream4, motorEncodeStream;
std::string base_path = "/home/wlp/Datasets/"; 
// 获取当前时间戳 (秒)
double get_time_sec() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000000.0;
}

void SigHandle(int sig)
{
    flg_exit = true;
    display_message("Catch signal " + std::to_string(sig) + ", exiting...");
    if(keyBoardCatch) {
        endwin();  // 结束ncurses模式
    }
    sig_buffer.notify_all();
}

// 解析IMU数据
IMUData parseIMUData(const uint8_t* buffer, int startIndex) {
    IMUData data;
    // 解析roll
    {
        unsigned char low_byte = buffer[startIndex + 1];
        unsigned char high_byte = buffer[startIndex + 2];
        int16_t value = (static_cast<int16_t>(high_byte) << 8) | low_byte;
        if (high_byte & 0x80) value |= 0xFFFF0000;
        data.roll = float(value) * 0.0001f * 180.0f / 3.1415926535;
    }
    // 解析pitch
    {
        unsigned char low_byte = buffer[startIndex + 3];
        unsigned char high_byte = buffer[startIndex + 4];
        int16_t value = (static_cast<int16_t>(high_byte) << 8) | low_byte;
        if (high_byte & 0x80) value |= 0xFFFF0000;
        data.pitch = float(value) * 0.0001f * 180.0f / 3.1415926535;
    }
    // 解析yaw
    {
        unsigned char low_byte = buffer[startIndex + 5];
        unsigned char high_byte = buffer[startIndex + 6];
        int16_t value = (static_cast<int16_t>(high_byte) << 8) | low_byte;
        if (high_byte & 0x80) value |= 0xFFFF0000;
        data.yaw = float(value) * 0.0001f * 180.0f / 3.1415926535;
    }
    // 解析acc_x
    {
        unsigned char low_byte = buffer[startIndex + 7];
        unsigned char high_byte = buffer[startIndex + 8];
        int16_t value = (static_cast<int16_t>(high_byte) << 8) | low_byte;
        if (high_byte & 0x80) value |= 0xFFFF0000;
        data.acc_x = float(value) * 0.01f;
    }
    // 解析acc_y
    {
        unsigned char low_byte = buffer[startIndex + 9];
        unsigned char high_byte = buffer[startIndex + 10];
        int16_t value = (static_cast<int16_t>(high_byte) << 8) | low_byte;
        if (high_byte & 0x80) value |= 0xFFFF0000;
        data.acc_y = float(value) * 0.01f;
    }
    // 解析acc_z
    {
        unsigned char low_byte = buffer[startIndex + 11];
        unsigned char high_byte = buffer[startIndex + 12];
        int16_t value = (static_cast<int16_t>(high_byte) << 8) | low_byte;
        if (high_byte & 0x80) value |= 0xFFFF0000;
        data.acc_z = float(value) * 0.01f;
    }
    // 解析gyro_x
    {
        unsigned char low_byte = buffer[startIndex + 13];
        unsigned char high_byte = buffer[startIndex + 14];
        int16_t value = (static_cast<int16_t>(high_byte) << 8) | low_byte;
        if (high_byte & 0x80) value |= 0xFFFF0000;
        data.gyro_x = float(value) * 0.001f * 180.0f / 3.1415926535;
    }
    // 解析gyro_y
    {
        unsigned char low_byte = buffer[startIndex + 15];
        unsigned char high_byte = buffer[startIndex + 16];
        int16_t value = (static_cast<int16_t>(high_byte) << 8) | low_byte;
        if (high_byte & 0x80) value |= 0xFFFF0000;
        data.gyro_y = float(value) * 0.001f * 180.0f / 3.1415926535;
    }
    // 解析gyro_z
    {
        unsigned char low_byte = buffer[startIndex + 17];
        unsigned char high_byte = buffer[startIndex + 18];
        int16_t value = (static_cast<int16_t>(high_byte) << 8) | low_byte;
        if (high_byte & 0x80) value |= 0xFFFF0000;
        data.gyro_z = float(value) * 0.001f * 180.0f / 3.1415926535;
    }
    return data;
}



// 发送力矩指令
// 如果需要在主循环中测试发送，调用此函数即可
uint8_t moment_buffer_send[10] = {0xA1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xAF};
void sendTorqueCommand(float left_moment, float right_moment) {
    // 右侧力矩取反
    float right_moment_send = -right_moment;

    memcpy(&moment_buffer_send[1], &left_moment, 4);
    memcpy(&moment_buffer_send[5], &right_moment_send, 4);

    if (serial_port >= 0) {
        int wrtSize = write(serial_port, moment_buffer_send, 10);
        std::cout << "write bytes: " << wrtSize << std::endl;
    }
}

// 变量记录初始时间
double first_imu_1_time = -1;
double first_imu_2_time = -1;
double first_imu_3_time = -1;
double first_imu_4_time = -1;

bool first_in = true;
bool find_new_head = false;
uint8_t buffer[256];
std::deque<uint8_t> buff_dy;

// 核心处理线程
void processIMU()
{
    while (!flg_exit)
    {
        memset(buffer, 0, sizeof(buffer));
        int bytes_read = read(serial_port, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 100) continue;
        
        if (first_in)
        {
            first_in = false;
            bool find_head = false;
            int data_i = 0;
            for (; data_i < bytes_read; data_i++) {
                if ((buffer[data_i] == IMU1_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU1_HEAD2)) ||
                    (buffer[data_i] == IMU2_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU2_HEAD2)) ||
                    (buffer[data_i] == IMU3_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU3_HEAD2)) ||
                    (buffer[data_i] == IMU4_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU4_HEAD2)) ||
                    (buffer[data_i] == MOTOR_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == MOTOR_HEAD2))) 
                {
                    find_head = true;
                    break;
                }
            }
            if (find_head) {
                display_message("Found header detected, starting sync...");
                bytes_read = 0; // 丢弃第一包剩余数据，等待完整帧
            }
        }

        if (bytes_read > 0)
        {
            for (int i = 0; i < bytes_read; ++i) {
                buff_dy.push_back(buffer[i]);
            }
        }

        while (buff_dy.size() >= 22) { 
            if (find_new_head)
            {
                find_new_head = false;
                while (!buff_dy.empty())
                {
                    // 简化的找头逻辑，与原代码一致
                    uint8_t h1 = buff_dy.front();
                    bool valid_head = false;
                    
                    if (buff_dy.size() > 1) {
                        uint8_t h2 = buff_dy[1];
                        if ((h1 == IMU1_HEAD1 && h2 == IMU1_HEAD2) ||
                            (h1 == IMU2_HEAD1 && h2 == IMU2_HEAD2) ||
                            (h1 == IMU3_HEAD1 && h2 == IMU3_HEAD2) ||
                            (h1 == IMU4_HEAD1 && h2 == IMU4_HEAD2) ||
                            (h1 == MOTOR_HEAD1 && h2 == MOTOR_HEAD2)) {
                            valid_head = true;
                        }
                    }
                    
                    if (valid_head) break;
                    buff_dy.pop_front();
                }
                if (buff_dy.size() < 22) break;
            }

            uint8_t temp_buffer[22];
            for (int i = 0; i < 22; ++i) {
                temp_buffer[i] = buff_dy.front();
                buff_dy.pop_front();
            }

            // --- IMU 1 ---
            if (temp_buffer[0] == IMU1_HEAD1 && temp_buffer[1] == IMU1_HEAD2) {
                if (temp_buffer[20] == IMU1_END1 && temp_buffer[21] == IMU1_END2) {
                    IMUData data = parseIMUData(temp_buffer, 1);
                    if (imu1_debug_out) {
                        display_message("[IMU1] AccX: " + std::to_string(data.acc_x) + " GyroZ: " + std::to_string(data.gyro_z));
                        imu1_debug_out = false;      
                    }
                    
                    {
                        std::lock_guard<std::mutex> lock(data_mutex);
                        global_data_buffer[6] = data.acc_x;
                        global_data_buffer[7] = data.acc_y;
                        global_data_buffer[8] = data.acc_z;
                        global_data_buffer[9] = data.gyro_x;
                        global_data_buffer[10] = data.gyro_y;
                        global_data_buffer[11] = data.gyro_z;
                    }

                    double timeNow = get_time_sec();
                    if (first_imu_1_time < 0) first_imu_1_time = timeNow;

                    imuEulerStream1.open(base_path + "imuData1.csv", std::ios::app);
                    imuEulerStream1 << std::fixed << std::setprecision(5) << timeNow << ","
                                    << first_imu_1_time << ","
                                    << data.roll << "," << data.pitch << "," << data.yaw << ","
                                    << data.gyro_x << "," << data.gyro_y << "," << data.gyro_z << ","
                                    << data.acc_x << "," << data.acc_y << "," << data.acc_z << ","
                                    << flag_num << std::endl;
                    imuEulerStream1.close();
                    first_imu_1_time += 0.01;
                }
                else {
                    // Header ok, Tail wrong -> rollback
                    for (int i = 21; i > 1; i--) buff_dy.push_front(temp_buffer[i]);
                    find_new_head = true; continue;
                }
            }
            // --- IMU 2 ---
            else if (temp_buffer[0] == IMU2_HEAD1 && temp_buffer[1] == IMU2_HEAD2) {
                if(temp_buffer[20] == IMU2_END1 && temp_buffer[21] == IMU2_END2) {
                    IMUData data = parseIMUData(temp_buffer, 1);
                    if (imu1_debug_out) imu1_debug_out = false; // Sync debug flag

                    {
                        std::lock_guard<std::mutex> lock(data_mutex);
                        global_data_buffer[12] = data.acc_x;
                        global_data_buffer[13] = data.acc_y;
                        global_data_buffer[14] = data.acc_z;
                        global_data_buffer[15] = data.gyro_x;
                        global_data_buffer[16] = data.gyro_y;
                        global_data_buffer[17] = data.gyro_z;
                    }

                    double timeNow = get_time_sec();
                    if (first_imu_2_time < 0) first_imu_2_time = timeNow;

                    imuEulerStream2.open(base_path + "imuData2.csv", std::ios::app);
                    imuEulerStream2 << std::fixed << std::setprecision(5) << timeNow << "," << first_imu_2_time << ","
                                     << data.roll << "," << data.pitch << "," << data.yaw << ","
                                     << data.gyro_x << "," << data.gyro_y << "," << data.gyro_z << ","
                                     << data.acc_x << "," << data.acc_y << "," << data.acc_z << ","
                                     << flag_num << std::endl;
                    imuEulerStream2.close();
                    first_imu_2_time += 0.01;
                }
                else {
                    for (int i = 21; i > 1; i--) buff_dy.push_front(temp_buffer[i]);
                    find_new_head = true; continue;
                }
            }
            // --- IMU 3 ---
            else if (temp_buffer[0] == IMU3_HEAD1 && temp_buffer[1] == IMU3_HEAD2) {
                if (temp_buffer[20] == IMU3_END1 && temp_buffer[21] == IMU3_END2) {
                    IMUData data = parseIMUData(temp_buffer, 1);
                    
                    double timeNow = get_time_sec();
                    if (first_imu_3_time < 0) first_imu_3_time = timeNow;

                    imuEulerStream3.open(base_path + "imuData3.csv", std::ios::app);
                    imuEulerStream3 << std::fixed << std::setprecision(5) << timeNow << "," << first_imu_3_time << ","
                                    << data.roll << "," << data.pitch << "," << data.yaw << ","
                                    << data.gyro_x << "," << data.gyro_y << "," << data.gyro_z << ","
                                    << data.acc_x << "," << data.acc_y << "," << data.acc_z << ","
                                    << flag_num << std::endl;
                    imuEulerStream3.close();
                    first_imu_3_time += 0.01;
                }
                else {
                    for (int i = 21; i > 1; i--) buff_dy.push_front(temp_buffer[i]);
                    find_new_head = true; continue;
                }
            }
            // --- IMU 4 ---
            else if (temp_buffer[0] == IMU4_HEAD1 && temp_buffer[1] == IMU4_HEAD2) {
                if (temp_buffer[20] == IMU4_END1 && temp_buffer[21] == IMU4_END2) {
                    IMUData data = parseIMUData(temp_buffer, 1);
                    
                    double timeNow = get_time_sec();
                    if (first_imu_4_time < 0) first_imu_4_time = timeNow;

                    imuEulerStream4.open(base_path + "imuData4.csv", std::ios::app);
                    imuEulerStream4 << std::fixed << std::setprecision(5) << timeNow << "," << first_imu_4_time << ","
                                    << data.roll << "," << data.pitch << "," << data.yaw << ","
                                    << data.gyro_x << "," << data.gyro_y << "," << data.gyro_z << ","
                                    << data.acc_x << "," << data.acc_y << "," << data.acc_z << ","
                                    << flag_num << std::endl;
                    imuEulerStream4.close();
                    first_imu_4_time += 0.01;
                }
                else {
                    for (int i = 21; i > 1; i--) buff_dy.push_front(temp_buffer[i]);
                    find_new_head = true; continue;
                }
            }
            // --- MOTOR ---
            else if (temp_buffer[0] == MOTOR_HEAD1 && temp_buffer[1] == MOTOR_HEAD2) {
                if (temp_buffer[20] == MOTOR_END1 && temp_buffer[21] == MOTOR_END2) {
                    uint16_t motor_L = (temp_buffer[2] << 8) | temp_buffer[3];
                    uint16_t motor_R = (temp_buffer[4] << 8) | temp_buffer[5];
                    
                    float motor_L_ = (float)(motor_L / 65535.0f * 360.0f); 
                    float motor_R_ = (float)(motor_R / 65535.0f * 360.0f);
                    // 弧度转换 + 中心化
                    motor_L_ = (motor_L_ - 180.0f) * 3.1415926535f / 180.0f;
                    motor_R_ = (motor_R_ - 180.0f) * 3.1415926535f / 180.0f;

                    int16_t motor_L_vel = (temp_buffer[6] << 8) | temp_buffer[7];
                    int16_t motor_R_vel = (temp_buffer[8] << 8) | temp_buffer[9];
                    // 速度转换 + 左侧反向
                    float motor_L_vel_ = -(float)(motor_L_vel / 36.0f); 
                    float motor_R_vel_ = (float)(motor_R_vel / 36.0f);
                    motor_L_vel_ = motor_L_vel_ * 3.1415926535f / 180.0f;
                    motor_R_vel_ = motor_R_vel_ * 3.1415926535f / 180.0f;

                    // 扭矩
                    int16_t motor_L_torque = (temp_buffer[10] << 8) | temp_buffer[11];
                    int16_t motor_R_torque = (temp_buffer[12] << 8) | temp_buffer[13];
                    float motor_L_torque_ = (float)(motor_L_torque);
                    float motor_R_torque_ = (float)(motor_R_torque);

                    {
                        std::lock_guard<std::mutex> lock(data_mutex);
                        global_data_buffer[0] = motor_L_;
                        global_data_buffer[1] = motor_R_;
                        global_data_buffer[2] = motor_L_vel_;
                        global_data_buffer[3] = motor_R_vel_;
                        global_data_buffer[18] = motor_L_torque_;
                        global_data_buffer[19] = motor_R_torque_;
                    }

                    if (motor_debug_out) {
                        display_message("[Motor] PosL: " + std::to_string(motor_L_) + " PosR: " + std::to_string(motor_R_));
                        motor_debug_out = false;
                    }

                    double timeNow = get_time_sec();
                    motorEncodeStream.open(base_path + "motorAngle.csv", std::ios::app);
                    motorEncodeStream << std::fixed << std::setprecision(5) << timeNow << ","
                                    << motor_L << "," << motor_L_ << ","
                                    << motor_R << "," << motor_R_ << ","
                                    << motor_L_vel << "," << motor_L_vel_ << ","
                                    << motor_R_vel << "," << motor_R_vel_ << ","
                                    << flag_num << std::endl;
                    motorEncodeStream.close();
                }
                else {
                    for (int i = 21; i > 1; i--) buff_dy.push_front(temp_buffer[i]);
                    find_new_head = true; continue;
                }
            }
            else {
                // All heads wrong
                display_message("Data corrupted, resyncing...");
                find_new_head = true;
            }
        }
    }
}

// 初始化CSV表头
void init_csv(std::string filename, bool is_imu) {
    std::fstream fs;
    fs.open(base_path + filename, std::ios::out);
    if (!fs.is_open()) {
        display_message("Cannot create file: " + base_path + filename);
        return;
    }
    if (is_imu) {
        fs <<  "timeStamp,timeStampAlign,roll,pitch,yaw,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z,flag_num" << std::endl;
    } else {
        fs <<  "timeStamp,angle_L_raw,angle_L_rad,angle_R_raw,angle_R_rad,vel_L_raw,vel_L_rad,vel_R_raw,vel_R_rad,flag_num" << std::endl;
    }
    fs.close();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <serial_port>" << std::endl;
        return 1;
    }

    const char *uartName = argv[1];
    // 初始化ncurses
    if(keyBoardCatch) {
        initscr();           // 初始化ncurses
        cbreak();            // 禁用行缓冲，立即获取输入
        noecho();            // 不显示输入字符
        nodelay(stdscr, TRUE); // 非阻塞模式
        keypad(stdscr, TRUE);  // 启用特殊键处理
        clear();             // 清屏
    }

    // Init CSV files
    init_csv("imuData1.csv", true);
    init_csv("imuData2.csv", true);
    init_csv("imuData3.csv", true);
    init_csv("imuData4.csv", true);
    init_csv("motorAngle.csv", false);

    // Serial Setup
    serial_port = open(uartName, O_RDWR);
    if (serial_port < 0) {
        display_message("Error opening serial port");
        return 1;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(serial_port, &tty) != 0) {
        std::cerr << "Error getting serial attributes" << std::endl;
        close(serial_port);
        return 1;
    }

    cfsetospeed(&tty, B921600); // 921600 波特率
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    // 禁用流控
    tty.c_cflag &= ~CRTSCTS;
    // 设置非规范模式 (Raw mode)
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); 
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~OPOST;

    tcsetattr(serial_port, TCSANOW, &tty);

    signal(SIGINT, SigHandle);

    // 启动接收线程
    std::thread imu_thread(processIMU);

    uint16_t debugOutCnt = 0;
    
    display_message("Driver started. Press Ctrl+C to exit.");

    // 主循环 (替代 ROS loop)
    while (!flg_exit) {
        
        // 模拟 100Hz 循环
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // 定时打印调试信息
        debugOutCnt++;
        if (debugOutCnt >= 10) {
            debugOutCnt = 0;
            imu1_debug_out = true;
            motor_debug_out = true;
            
            // 示例：访问全局数据
            std::lock_guard<std::mutex> lock(data_mutex);
            display_message("Current Motor L Rad: " + std::to_string(global_data_buffer[0]));
        }

        // 捕捉键盘输入
        if (keyBoardCatch)
        {
            // 捕捉键盘
            int ch_ = getch();
            // 检查输入是否是数字
            if (ch_ >= '0' && ch_ <= '9') {
                flag_num = ch_ - '0';
                display_message("press num: " + std::to_string(flag_num));
            } 
            else if (ch_ != ERR) {  // ERR means no input
                // 如果按了其他键，也可以处理
                // 例如按 q 退出
                if (ch_ == 'q' || ch_ == 'Q') {
                    flg_exit = true;
                }
            }
        }

        // TODO: 如果需要发送力矩，可以在这里调用 sendTorqueCommand(0.0, 0.0);
    }

    close(serial_port);
    if (imu_thread.joinable()) {
        imu_thread.join();
    }
    if(keyBoardCatch) {
        endwin();  // 结束ncurses模式
    }
    std::cout << "Program finished." << std::endl;
    return 0;
}