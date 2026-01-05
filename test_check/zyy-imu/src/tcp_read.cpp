#include <iostream>
#include <cstring> // for memset, memcpy
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>  // 用于时间戳
#include <fstream>
#include <queue>
#include <deque>
#include <csignal> // for SIGINT
#include <thread>
#include <mutex>
#include <Eigen/Dense>
#include <Eigen/Geometry>  // 用于四元数操作

// --- Linux 专用网络头文件 ---
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h> // close, read, write
#include <errno.h>  // errno
// --- 结束 Linux 头文件 ---

// --- 兼容性定义 ---
typedef int SOCKET;
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
// -----------------

// 四元数转换函数 (保留)
Eigen::Quaternionf rightHandToLeftHandQuat(const Eigen::Quaternionf& q_right) {
    // 绕 X 轴旋转 180° 对应四元数为 [1, 0, 0, 0]
    Eigen::Quaternionf q_fix(0.0f, 1.0f, 0.0f, 0.0f);  // xyzw
    return q_fix * q_right;
}

Eigen::Quaterniond rhToLh(const Eigen::Quaterniond& qa) {
    Eigen::Matrix3d M = Eigen::Matrix3d::Identity();
    M(1,1) = -1;
    Eigen::Matrix3d R_qa = qa.toRotationMatrix();
    Eigen::Matrix3d R_qb = M * R_qa * M.transpose();
    Eigen::Quaterniond qb(R_qb);
    qb.normalize();
    return qb;
}

// 调试标志位 (保留)
bool imu1_debug_out = 0;
bool imu2_debug_out = 0;
bool imu3_debug_out = 0;
bool imu4_debug_out = 0;
bool imu5_debug_out = 0;
bool imu6_debug_out = 0;
bool imu7_debug_out = 0;
bool imu8_debug_out = 0;
bool imu9_debug_out = 0;
bool imu10_debug_out = 0;
bool imu11_debug_out = 0;
bool imu12_debug_out = 0;
bool imu13_debug_out = 0;

#define PORT 8086
#define BUFFER_SIZE 512

// 定义IMU的帧头帧尾 (保留)
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

#define IMU5_HEAD1 0x7b
#define IMU5_HEAD2 0xf5
#define IMU5_END1  0xe4
#define IMU5_END2  0xeb

#define IMU6_HEAD1 0x7a
#define IMU6_HEAD2 0xf6
#define IMU6_END1  0xe5
#define IMU6_END2  0xea

#define IMU7_HEAD1 0x79
#define IMU7_HEAD2 0xf7
#define IMU7_END1  0xe6
#define IMU7_END2  0xe9

#define IMU8_HEAD1 0x78
#define IMU8_HEAD2 0xf8
#define IMU8_END1  0xe7
#define IMU8_END2  0xe8

#define IMU9_HEAD1 0x77
#define IMU9_HEAD2 0xf9
#define IMU9_END1  0xe8
#define IMU9_END2  0xe7

#define IMU10_HEAD1 0x76
#define IMU10_HEAD2 0xfa
#define IMU10_END1  0xe9
#define IMU10_END2  0xe6

#define IMU11_HEAD1 0x75
#define IMU11_HEAD2 0xfb
#define IMU11_END1  0xea
#define IMU11_END2  0xe5

#define IMU12_HEAD1 0x74
#define IMU12_HEAD2 0xfc
#define IMU12_END1  0xeb
#define IMU12_END2  0xe4

#define IMU13_HEAD1 0x73
#define IMU13_HEAD2 0xfd
#define IMU13_END1  0xec
#define IMU13_END2  0xe3

// 信号处理
bool flg_exit = false;
void SigHandle(int sig)
{
    flg_exit = true;
    std::cerr << "Caught signal " << sig << ", shutting down." << std::endl;
}

// 时间戳标志 (保留)
double first_imu_1_time = -1;
double first_imu_2_time = -1;
double first_imu_3_time = -1;
double first_imu_4_time = -1;
double first_imu_5_time = -1;
double first_imu_6_time = -1;
double first_imu_7_time = -1;
double first_imu_8_time = -1;
double first_imu_9_time = -1;
double first_imu_10_time = -1;
double first_imu_11_time = -1;
double first_imu_12_time = -1;
double first_imu_13_time = -1;

// CSV 文件路径
std::string imuEulerFormat1 = "./imuData1.csv";
std::fstream imuEulerStream1;
std::string imuEulerFormat2 = "./imuData2.csv";
std::fstream imuEulerStream2;
std::string imuEulerFormat3 = "./imuData3.csv";
std::fstream imuEulerStream3;
std::string imuEulerFormat4 = "./imuData4.csv";
std::fstream imuEulerStream4;
std::string imuEulerFormat5 = "./imuData5.csv";
std::fstream imuEulerStream5;
std::string imuEulerFormat6 = "./imuData6.csv";
std::fstream imuEulerStream6;
std::string imuEulerFormat7 = "./imuData7.csv";
std::fstream imuEulerStream7;
std::string imuEulerFormat8 = "./imuData8.csv";
std::fstream imuEulerStream8;
std::string imuEulerFormat9 = "./imuData9.csv";
std::fstream imuEulerStream9;
std::string imuEulerFormat10 = "./imuData10.csv";
std::fstream imuEulerStream10;
std::string imuEulerFormat11 = "./imuData11.csv";
std::fstream imuEulerStream11;
std::string imuEulerFormat12 = "./imuData12.csv";
std::fstream imuEulerStream12;
std::string imuEulerFormat13 = "./imuData13.csv";
std::fstream imuEulerStream13;

bool first_in = true;
bool find_new_head = false;
std::deque<uint8_t> buff_dy;

char buffer[BUFFER_SIZE] = {0};
SOCKET new_socket = INVALID_SOCKET; // 全局 Socket

// ====== 动捕 UDP 控制 ======
void send_xml_capture_start(const std::string& target_ip, int target_port, const std::string& name_value)
{
    // 注意：DatabasePath 是 Windows 路径，如果接收方是 Windows 软件则无需修改
    std::string xml_message =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>\n"
        "<CaptureStart><Name VALUE=\"" + name_value + "\"/>\n"
        "<SessionName VALUE=\"SessionName\" />\n"
        "<Notes VALUE=\"Take notes if any\" />\n"
        "<Delay VALUE=\"Reserved\" /><Description VALUE=\"\" />\n"
        "<DatabasePath VALUE=\"D:/desktop/20250924\" />\n"
        "<TimeCode VALUE=\"00:00:00:00\"/><PacketID VALUE=\"0\"/>\n"
        "</CaptureStart>";

    // [Linux] 创建 UDP socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "Failed to create UDP socket" << std::endl;
        return;
    }

    sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(target_port);
    if (inet_pton(AF_INET, target_ip.c_str(), &dest.sin_addr) <= 0) {
        std::cerr << "Invalid address / Address not supported" << std::endl;
        close(sock);
        return;
    }

    sendto(sock, xml_message.c_str(), xml_message.size(), 0,
           (sockaddr*)&dest, sizeof(dest));

    close(sock); // [Linux] 使用 close

    std::cout << "\n===============================" << std::endl;
    std::cout << ">>> 动捕 CaptureStart 已发送！" << std::endl;
    std::cout << ">>> 名称：" << name_value << std::endl;
    std::cout << "===============================\n" << std::endl;
}

// 数据处理线程
void processIMU()
{
    while (!flg_exit)
    {
        memset(buffer, 0, sizeof(buffer));
        
        // [Linux] recv 返回 ssize_t (int 也可以), 出错返回 -1
        int bytes_read = recv(new_socket, buffer, BUFFER_SIZE, 0);

        if (bytes_read < 0) {
            // [Linux] 使用 errno 获取错误信息
            std::cerr << "recv failed: " << strerror(errno) << std::endl;
            break; 
        }

        if (bytes_read == 0) {
            std::cout << "Client disconnected gracefully." << std::endl;
            break; // 客户端主动关闭链接
        }

        // ... (数据缓冲和帧头查找逻辑保持不变) ...
        if (first_in)
        {
            first_in = false;
            bool find_head = false;
            int data_i = 0;
            for (; data_i < bytes_read; data_i++)
            {
                if (buffer[data_i] == IMU1_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU1_HEAD2))
                { find_head = true; break; }
                else if (buffer[data_i] == IMU2_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU2_HEAD2))
                { find_head = true; break; }
                else if (buffer[data_i] == IMU3_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU3_HEAD2))
                { find_head = true; break; }
                else if (buffer[data_i] == IMU4_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU4_HEAD2))
                { find_head = true; break; }
                if (buffer[data_i] == IMU5_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU5_HEAD2))
                { find_head = true; break; }
                else if (buffer[data_i] == IMU6_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU6_HEAD2))
                { find_head = true; break; }
                else if (buffer[data_i] == IMU7_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU7_HEAD2))
                { find_head = true; break; }
                else if (buffer[data_i] == IMU8_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU8_HEAD2))
                { find_head = true; break; }
                if (buffer[data_i] == IMU9_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU9_HEAD2))
                { find_head = true; break; }
                else if (buffer[data_i] == IMU10_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU10_HEAD2))
                { find_head = true; break; }
                else if (buffer[data_i] == IMU11_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU11_HEAD2))
                { find_head = true; break; }
                else if (buffer[data_i] == IMU12_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU12_HEAD2))
                { find_head = true; break; }
                else if (buffer[data_i] == IMU13_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == IMU13_HEAD2))
                { find_head = true; break; }
            }
            if (find_head)
            {
                std::ostringstream oss;
                for (; data_i < bytes_read; data_i++)
                {
                    buff_dy.push_back(buffer[data_i]);
                    oss << "0x" << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(buffer[data_i])) << " ";
                }
                std::cout << "first Received (hex): " << bytes_read << ":-> " << oss.str() << std::endl;
                bytes_read = 0;
            }
        }
        if (bytes_read > 0)
        {
            for (int i = 0; i < bytes_read; ++i) {
                buff_dy.push_back(buffer[i]);
            }
        }

         while (buff_dy.size() >= 36) { // 24 = 2 (head) + 20 (data) + 2 (tail)
            if (find_new_head)
            {
                find_new_head = false;
                while (!buff_dy.empty())
                {
                    if (buff_dy.front() == IMU1_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU1_HEAD2) { buff_dy.push_front(IMU1_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU2_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU2_HEAD2) { buff_dy.push_front(IMU2_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU3_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU3_HEAD2) { buff_dy.push_front(IMU3_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU4_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU4_HEAD2) { buff_dy.push_front(IMU4_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU5_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU5_HEAD2) { buff_dy.push_front(IMU5_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU6_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU6_HEAD2) { buff_dy.push_front(IMU6_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU7_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU7_HEAD2) { buff_dy.push_front(IMU7_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU8_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU8_HEAD2) { buff_dy.push_front(IMU8_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU9_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU9_HEAD2) { buff_dy.push_front(IMU9_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU10_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU10_HEAD2) { buff_dy.push_front(IMU10_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU11_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU11_HEAD2) { buff_dy.push_front(IMU11_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU12_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU12_HEAD2) { buff_dy.push_front(IMU12_HEAD1); break; } } else break;
                    }
                    else if (buff_dy.front() == IMU13_HEAD1) {
                        if (buff_dy.size() > 1) { buff_dy.pop_front(); if (buff_dy.front() == IMU13_HEAD2) { buff_dy.push_front(IMU13_HEAD1); break; } } else break;
                    }
                    buff_dy.pop_front();
                }
                if (buff_dy.size() < 36)
                    break;
            }

            uint8_t temp_buffer[36];
            for (int i = 0; i < 36; ++i) {
                temp_buffer[i] = buff_dy.front();
                buff_dy.pop_front();
            }

            auto stdTime_ = std::chrono::system_clock::now();
            double rosTimeNow = std::chrono::duration<double>(stdTime_.time_since_epoch()).count();

            if (temp_buffer[0] == IMU1_HEAD1 && temp_buffer[1] == IMU1_HEAD2) {
                if (temp_buffer[34] == IMU1_END1 && temp_buffer[35] == IMU1_END2) {
                    uint32_t idx;
                    float q_x, q_y, q_z, q_w, acc_x, acc_y, acc_z;
                    memcpy(&idx, &temp_buffer[2], 4);
                    memcpy(&q_x, &temp_buffer[6], 4);
                    memcpy(&q_y, &temp_buffer[10], 4);
                    memcpy(&q_z, &temp_buffer[14], 4);
                    memcpy(&q_w, &temp_buffer[18], 4);
                    memcpy(&acc_x, &temp_buffer[22], 4);
                    memcpy(&acc_y, &temp_buffer[26], 4);
                    memcpy(&acc_z, &temp_buffer[30], 4);
                    {
                        Eigen::Quaternionf qa1(q_w, q_x, q_y, q_z);
                        Eigen::Quaternionf qb1 = rightHandToLeftHandQuat(qa1);
                        q_x = qb1.x(); q_y = qb1.y(); q_z = qb1.z(); q_w = qb1.w();
                    }
                    if (imu1_debug_out)
                    {
                        std::cout << "imu1 idx: " << idx << ", q_x: " << q_x << ", q_y: " << q_y << std::endl;
                        imu1_debug_out = false;      
                    }

                    if (first_imu_1_time < 0) { first_imu_1_time = rosTimeNow; }
                    
                    imuEulerStream1.open(imuEulerFormat1, std::ios::app);
                    imuEulerStream1 << std::setiosflags(std::ios::fixed) << std::setprecision(5) << rosTimeNow << ","
                                    << idx << ","
                                    << std::setiosflags(std::ios::fixed) << std::setprecision(5) 
                                    << q_x << "," << q_y << "," << q_z << "," << q_w << ","
                                    << acc_x << "," << acc_y << "," << acc_z << std::endl;
                    imuEulerStream1.close();

                    // [REMOVED] ROS Publishing code
                }
                else {
                    for (int i = 35; i > 1; i--) { buff_dy.push_front(temp_buffer[i]); }
                    find_new_head = true;
                    continue;
                }
            }
            // ... (其他 IMU2 到 IMU13 的 'else if' 块) ...
            // ... (逻辑与 IMU1 相同, 只需移除 ROS 代码) ...

            else if (temp_buffer[0] == IMU2_HEAD1 && temp_buffer[1] == IMU2_HEAD2)
            {
                if(temp_buffer[34] == IMU2_END1 && temp_buffer[35] == IMU2_END2) {
                    uint32_t idx;
                    float q_x, q_y, q_z, q_w, acc_x, acc_y, acc_z;
                    memcpy(&idx, &temp_buffer[2], 4);
                    memcpy(&q_x, &temp_buffer[6], 4);
                    memcpy(&q_y, &temp_buffer[10], 4);
                    memcpy(&q_z, &temp_buffer[14], 4);
                    memcpy(&q_w, &temp_buffer[18], 4);
                    memcpy(&acc_x, &temp_buffer[22], 4);
                    memcpy(&acc_y, &temp_buffer[26], 4);
                    memcpy(&acc_z, &temp_buffer[30], 4);
                    {
                        Eigen::Quaternionf qa1(q_w, q_x, q_y, q_z);
                        Eigen::Quaternionf qb1 = rightHandToLeftHandQuat(qa1);
                        q_x = qb1.x(); q_y = qb1.y(); q_z = qb1.z(); q_w = qb1.w();
                    }
                    if (imu2_debug_out) {
                        std::cout << "imu2 idx: " << idx << std::endl;
                        imu2_debug_out = false;
                    }
                    if (first_imu_2_time < 0) { first_imu_2_time = rosTimeNow; }
                    imuEulerStream2.open(imuEulerFormat2, std::ios::app);
                    imuEulerStream2  << std::setiosflags(std::ios::fixed) << std::setprecision(5) << rosTimeNow << ","
                                    << idx << ","
                                    << std::setiosflags(std::ios::fixed) << std::setprecision(5) 
                                    << q_x << "," << q_y << "," << q_z << "," << q_w << ","
                                    << acc_x << "," << acc_y << "," << acc_z << std::endl;
                    imuEulerStream2.close();
                    // [REMOVED] ROS Publishing code
                }
                else {
                    for (int i = 35; i > 1; i--) { buff_dy.push_front(temp_buffer[i]); }
                    find_new_head = true;
                    continue;
                }
            }

            else if (temp_buffer[0] == IMU3_HEAD1 && temp_buffer[1] == IMU3_HEAD2)
            {
                if (temp_buffer[34] == IMU3_END1 && temp_buffer[35] == IMU3_END2) {
                    uint32_t idx;
                    float q_x, q_y, q_z, q_w, acc_x, acc_y, acc_z;
                    memcpy(&idx, &temp_buffer[2], 4);
                    memcpy(&q_x, &temp_buffer[6], 4);
                    memcpy(&q_y, &temp_buffer[10], 4);
                    memcpy(&q_z, &temp_buffer[14], 4);
                    memcpy(&q_w, &temp_buffer[18], 4);
                    memcpy(&acc_x, &temp_buffer[22], 4);
                    memcpy(&acc_y, &temp_buffer[26], 4);
                    memcpy(&acc_z, &temp_buffer[30], 4);
                    {
                        Eigen::Quaternionf qa1(q_w, q_x, q_y, q_z);
                        Eigen::Quaternionf qb1 = rightHandToLeftHandQuat(qa1);
                        q_x = qb1.x(); q_y = qb1.y(); q_z = qb1.z(); q_w = qb1.w();
                    }
                    if (imu3_debug_out) {
                        std::cout << "imu3 idx: " << idx << std::endl;
                        imu3_debug_out = false;
                    }
                    if (first_imu_3_time < 0) { first_imu_3_time = rosTimeNow; }
                    imuEulerStream3.open(imuEulerFormat3, std::ios::app);
                    imuEulerStream3 << std::setiosflags(std::ios::fixed) << std::setprecision(5) << rosTimeNow << ","
                                    << idx << ","
                                    << std::setiosflags(std::ios::fixed) << std::setprecision(5) 
                                    << q_x << "," << q_y << "," << q_z << "," << q_w << ","
                                    << acc_x << "," << acc_y << "," << acc_z << std::endl;
                    imuEulerStream3.close();
                    // [REMOVED] ROS Publishing code
                }
                else {
                    for (int i = 35; i > 1; i--) { buff_dy.push_front(temp_buffer[i]); }
                    find_new_head = true;
                    continue;
                }
            }

            else if (temp_buffer[0] == IMU4_HEAD1 && temp_buffer[1] == IMU4_HEAD2)
            {
                if (temp_buffer[34] == IMU4_END1 && temp_buffer[35] == IMU4_END2) {
                    uint32_t idx;
                    float q_x, q_y, q_z, q_w, acc_x, acc_y, acc_z;
                    memcpy(&idx, &temp_buffer[2], 4);
                    memcpy(&q_x, &temp_buffer[6], 4);
                    memcpy(&q_y, &temp_buffer[10], 4);
                    memcpy(&q_z, &temp_buffer[14], 4);
                    memcpy(&q_w, &temp_buffer[18], 4);
                    memcpy(&acc_x, &temp_buffer[22], 4);
                    memcpy(&acc_y, &temp_buffer[26], 4);
                    memcpy(&acc_z, &temp_buffer[30], 4);
                    {
                        Eigen::Quaternionf qa1(q_w, q_x, q_y, q_z);
                        Eigen::Quaternionf qb1 = rightHandToLeftHandQuat(qa1);
                        q_x = qb1.x(); q_y = qb1.y(); q_z = qb1.z(); q_w = qb1.w();
                    }
                    if (imu4_debug_out) {
                        std::cout << "imu4 idx: " << idx << std::endl;
                        imu4_debug_out = false;
                    }
                    if (first_imu_4_time < 0) { first_imu_4_time = rosTimeNow; }
                    imuEulerStream4.open(imuEulerFormat4, std::ios::app);
                    imuEulerStream4 << std::setiosflags(std::ios::fixed) << std::setprecision(5) << rosTimeNow << ","
                                    << idx << ","
                                    << std::setiosflags(std::ios::fixed) << std::setprecision(5) 
                                    << q_x << "," << q_y << "," << q_z << "," << q_w << ","
                                    << acc_x << "," << acc_y << "," << acc_z << std::endl;
                    imuEulerStream4.close();
                    // [REMOVED] ROS Publishing code
                }
                else {
                    for (int i = 35; i > 1; i--) { buff_dy.push_front(temp_buffer[i]); }
                    find_new_head = true;
                    continue;
                }
            }

            else if (temp_buffer[0] == IMU5_HEAD1 && temp_buffer[1] == IMU5_HEAD2)
            {
                if(temp_buffer[34] == IMU5_END1 && temp_buffer[35] == IMU5_END2) {
                    uint32_t idx;
                    float q_x, q_y, q_z, q_w, acc_x, acc_y, acc_z;
                    memcpy(&idx, &temp_buffer[2], 4);
                    memcpy(&q_x, &temp_buffer[6], 4);
                    memcpy(&q_y, &temp_buffer[10], 4);
                    memcpy(&q_z, &temp_buffer[14], 4);
                    memcpy(&q_w, &temp_buffer[18], 4);
                    memcpy(&acc_x, &temp_buffer[22], 4);
                    memcpy(&acc_y, &temp_buffer[26], 4);
                    memcpy(&acc_z, &temp_buffer[30], 4);
                    {
                        Eigen::Quaternionf qa1(q_w, q_x, q_y, q_z);
                        Eigen::Quaternionf qb1 = rightHandToLeftHandQuat(qa1);
                        q_x = qb1.x(); q_y = qb1.y(); q_z = qb1.z(); q_w = qb1.w();
                    }
                    if (imu5_debug_out) {
                        std::cout << "imu5 idx: " << idx << std::endl;
                        imu5_debug_out = false;
                    }
                    if (first_imu_5_time < 0) { first_imu_5_time = rosTimeNow; }
                    imuEulerStream5.open(imuEulerFormat5, std::ios::app);
                    imuEulerStream5  << std::setiosflags(std::ios::fixed) << std::setprecision(5) << rosTimeNow << ","
                                    << idx << ","
                                    << std::setiosflags(std::ios::fixed) << std::setprecision(5) 
                                    << q_x << "," << q_y << "," << q_z << "," << q_w << ","
                                    << acc_x << "," << acc_y << "," << acc_z << std::endl;
                    imuEulerStream5.close();
                    // [REMOVED] ROS Publishing code
                }
                else {
                    for (int i = 35; i > 1; i--) { buff_dy.push_front(temp_buffer[i]); }
                    find_new_head = true;
                    continue;
                }
            }

            else if (temp_buffer[0] == IMU6_HEAD1 && temp_buffer[1] == IMU6_HEAD2)
            {
                if(temp_buffer[34] == IMU6_END1 && temp_buffer[35] == IMU6_END2) {
                    uint32_t idx;
                    float q_x, q_y, q_z, q_w, acc_x, acc_y, acc_z;
                    memcpy(&idx, &temp_buffer[2], 4);
                    memcpy(&q_x, &temp_buffer[6], 4);
                    memcpy(&q_y, &temp_buffer[10], 4);
                    memcpy(&q_z, &temp_buffer[14], 4);
                    memcpy(&q_w, &temp_buffer[18], 4);
                    memcpy(&acc_x, &temp_buffer[22], 4);
                    memcpy(&acc_y, &temp_buffer[26], 4);
                    memcpy(&acc_z, &temp_buffer[30], 4);
                    {
                        Eigen::Quaternionf qa1(q_w, q_x, q_y, q_z);
                        Eigen::Quaternionf qb1 = rightHandToLeftHandQuat(qa1);
                        q_x = qb1.x(); q_y = qb1.y(); q_z = qb1.z(); q_w = qb1.w();
                    }
                    if (imu6_debug_out) {
                        std::cout << "imu6 idx: " << idx << std::endl;
                        imu6_debug_out = false;
                    }
                    if (first_imu_6_time < 0) { first_imu_6_time = rosTimeNow; }
                    imuEulerStream6.open(imuEulerFormat6, std::ios::app);
                    imuEulerStream6  << std::setiosflags(std::ios::fixed) << std::setprecision(5) << rosTimeNow << ","
                                    << idx << ","
                                    << std::setiosflags(std::ios::fixed) << std::setprecision(5) 
                                    << q_x << "," << q_y << "," << q_z << "," << q_w << ","
                                    << acc_x << "," << acc_y << "," << acc_z << std::endl;
                    imuEulerStream6.close();
                    // [REMOVED] ROS Publishing code
                }
                else {
                    for (int i = 35; i > 1; i--) { buff_dy.push_front(temp_buffer[i]); }
                    find_new_head = true;
                    continue;
                }
            }
            
            else
            {
                // 简单的防刷屏
                static int wrong_head_count = 0;
                wrong_head_count++;
                if(wrong_head_count % 100 == 0) {
                     std::cout << "finding head..." << std::endl;
                }
                find_new_head = true;
            }
        }
    }
    std::cout << "Data processing thread exiting." << std::endl;
}


int main(int argc, char ** argv) {

    // [Linux Change] 移除 WSAStartup

    imuEulerStream1.open(imuEulerFormat1, std::ios::out); imuEulerStream1.close();
    imuEulerStream1.open(imuEulerFormat1, std::ios::app);
    imuEulerStream1 << "timeStamp" << "," << "idx" << "," << "q_x" << "," << "q_y" << "," << "q_z" << "," << "q_w" << "," << "acc_x" << "," << "acc_y" << "," << "acc_z" << "," << std::endl;
    imuEulerStream1.close();
    
    imuEulerStream2.open(imuEulerFormat2, std::ios::out); imuEulerStream2.close();
    imuEulerStream2.open(imuEulerFormat2, std::ios::app);
    imuEulerStream2 << "timeStamp" << "," << "idx" << "," << "q_x" << "," << "q_y" << "," << "q_z" << "," << "q_w" << "," << "acc_x" << "," << "acc_y" << "," << "acc_z" << "," << std::endl;
    imuEulerStream2.close();

    imuEulerStream3.open(imuEulerFormat3, std::ios::out); imuEulerStream3.close();
    imuEulerStream3.open(imuEulerFormat3, std::ios::app);
    imuEulerStream3 << "timeStamp" << "," << "idx" << "," << "q_x" << "," << "q_y" << "," << "q_z" << "," << "q_w" << "," << "acc_x" << "," << "acc_y" << "," << "acc_z" << "," << std::endl;
    imuEulerStream3.close();

    imuEulerStream4.open(imuEulerFormat4, std::ios::out); imuEulerStream4.close();
    imuEulerStream4.open(imuEulerFormat4, std::ios::app);
    imuEulerStream4 << "timeStamp" << "," << "idx" << "," << "q_x" << "," << "q_y" << "," << "q_z" << "," << "q_w" << "," << "acc_x" << "," << "acc_y" << "," << "acc_z" << "," << std::endl;
    imuEulerStream4.close();

    imuEulerStream5.open(imuEulerFormat5, std::ios::out); imuEulerStream5.close();
    imuEulerStream5.open(imuEulerFormat5, std::ios::app);
    imuEulerStream5 << "timeStamp" << "," << "idx" << "," << "q_x" << "," << "q_y" << "," << "q_z" << "," << "q_w" << "," << "acc_x" << "," << "acc_y" << "," << "acc_z" << "," << std::endl;
    imuEulerStream5.close();

    imuEulerStream6.open(imuEulerFormat6, std::ios::out); imuEulerStream6.close();
    imuEulerStream6.open(imuEulerFormat6, std::ios::app);
    imuEulerStream6 << "timeStamp" << "," << "idx" << "," << "q_x" << "," << "q_y" << "," << "q_z" << "," << "q_w" << "," << "acc_x" << "," << "acc_y" << "," << "acc_z" << "," << std::endl;
    imuEulerStream6.close();



    // [Linux Change] Socket 设置
    SOCKET server_fd = INVALID_SOCKET;
    struct sockaddr_in address;
    int opt = 1;
    socklen_t addrlen = sizeof(address); // Linux 使用 socklen_t

    // 创建套接字
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 设置套接字选项 (Linux下 setsockopt 第4个参数是 void*, 不需要转 const char*)
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // 绑定
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // 监听
    if (listen(server_fd, 3) < 0) {
        perror("listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    std::cout << "Server is listening on port " << PORT << std::endl;

    // 接受连接
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen)) < 0) {
        perror("accept failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // 接受一个连接后就关闭监听套接字
    close(server_fd); // [Linux] 使用 close

    std::cout << "Connection accepted." << std::endl;

    // ====== 加入动捕开始触发 ======
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&tt), "%Y%m%d_%H%M%S");
    std::string timestamp_name = ss.str();

    // 发送触发指令
    send_xml_capture_start("10.1.1.198", 7060, timestamp_name);
    // ===================================

    // 设置信号处理
    signal(SIGINT, SigHandle);

    // 启动数据处理线程
    std::thread imu_thread(processIMU);

    // 等待数据处理线程结束 (当客户端断开或按 Ctrl+C 时)
    imu_thread.join();

    // 清理
    std::cout << "Shutting down." << std::endl;
    close(new_socket); // [Linux] 使用 close
    // [Linux Change] 移除 WSACleanup
    return 0;
}