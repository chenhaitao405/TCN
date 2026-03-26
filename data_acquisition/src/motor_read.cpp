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
#include <ctime>
#include <atomic>
#include <cstdint>
#include <algorithm> 
#include <numeric>   

// 全局变量控制退出
std::atomic<bool> flg_exit{false};
std::atomic<bool> cleanup_done{false};  // 防止重复清理
std::condition_variable sig_buffer;
bool keyBoardCatch = true; 

// 串口相关定义
#define SENSOR_HEAD1 0x7e
#define SENSOR_HEAD2 0xf2
#define SENSOR_END1  0xe1
#define SENSOR_END2  0xee
#define SENSOR_FRAME_SIZE 53

// 定义传感器数据结构
struct SensorData {
    float acc_x, acc_y, acc_z;
    float gyro_x, gyro_y, gyro_z;
    float motor_angle_L, motor_angle_R;
    float motor_vel_L, motor_vel_R;
    uint8_t pattern_result;
    float torque;
    float Confidence;
};

// 全局数据存储
std::vector<float> global_data_buffer(13, 0.0f);
std::mutex data_mutex; 

// 历史数据存储
struct DataRecord {
    float acc[3];
    float gyro[3];
    float motor_ang[2];
    float motor_vel[2];
    uint8_t pattern_result;
    float torque;
    float confidence;
    double timestamp;
};
std::deque<DataRecord> data_history;
const int MAX_HISTORY = 8;
std::mutex history_mutex;

// === 新增：零偏校准相关全局变量 ===
std::atomic<bool> is_calibrated{false}; // 是否完成校准
float offset_motor_L = 0.0f;            // 左电机偏置
float offset_motor_R = 0.0f;            // 右电机偏置
const int CALIB_SAMPLES = 50;           // 校准所需的样本数（若频率为100Hz，50个样本约0.5秒）
const float STABILITY_THRESHOLD = 0.05f;// 波动阈值
std::vector<float> calib_buf_L;
std::vector<float> calib_buf_R;
// ==================================

// 全局配置
std::atomic<int> flag_num{0};
bool sensor_debug_out = 0;
int serial_port;

// flag=2 累积统计
std::atomic<uint64_t> total_flag2_samples{0};
std::atomic<uint64_t> total_pattern2_samples{0};

// 文件流
std::fstream dataStream;
std::string base_path = "/home/wlp/Datasets/"; 
std::string file_path_data;

// === 终端显示函数（普通追加模式） ===
void display_status(const std::string& msg) {
    std::cout << "[Status] " << msg << std::endl;
}

void display_message(const std::string& msg) {
    std::cout << "[Message] " << msg << std::endl;
}

// 打印最新一条传感器数据，终端会自然向下滚动
void display_history_status() {
    std::lock_guard<std::mutex> lock(history_mutex);
    if (!data_history.empty()) {
        auto& latest = data_history.back();
        std::cout << std::fixed << std::setprecision(3) 
                  << "[Time: " << std::setw(8) << latest.timestamp << "] "
                  << "Acc: " << std::setw(6) << latest.acc[0] << "," << std::setw(6) << latest.acc[1] << "," << std::setw(6) << latest.acc[2] << " | "
                  << "Gyro: " << std::setw(6) << latest.gyro[0] << "," << std::setw(6) << latest.gyro[1] << "," << std::setw(6) << latest.gyro[2] << " | "
                  << "MotL: " << std::setw(6) << latest.motor_ang[0] << "(" << std::setw(6) << latest.motor_vel[0] << ") | "
                  << "MotR: " << std::setw(6) << latest.motor_ang[1] << "(" << std::setw(6) << latest.motor_vel[1] << ") | "
                  << "Pattern: " << std::setw(3) << static_cast<int>(latest.pattern_result) << " | "
                  << "Torque: " << std::setw(6) << latest.torque << " | "
                  << "Conf: " << std::setw(6) << latest.confidence << " | "
                  << "Flag: " << flag_num.load()
                  << std::endl;
    }
}

// 获取当前日期时间字符串 YYYYMMDD_HHMMSS
std::string get_current_datetime_str() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
    return ss.str();
}

// 获取当前时间戳 (秒)
double get_time_sec() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000000.0;
}

// 信号处理 (Ctrl+C 等)
void SigHandle(int sig) {
    if (cleanup_done.load()) return;
    flg_exit.store(true);
    
    std::string sig_name;
    switch(sig) {
        case SIGINT:  sig_name = "SIGINT(Ctrl+C)"; break;
        case SIGTSTP: sig_name = "SIGTSTP(Ctrl+Z)"; break;
        default:      sig_name = "signal " + std::to_string(sig); break;
    }
    std::cout << "\n>>> Caught " << sig_name << ", exiting gracefully... <<<" << std::endl;
    sig_buffer.notify_all();
}

// 解析传感器数据
SensorData parseSensorData(const uint8_t* buffer) {
    SensorData data{};
    memcpy(&data.acc_x, &buffer[2], 4);
    memcpy(&data.acc_y, &buffer[6], 4);
    memcpy(&data.acc_z, &buffer[10], 4);
    memcpy(&data.gyro_x, &buffer[14], 4);
    memcpy(&data.gyro_y, &buffer[18], 4);
    memcpy(&data.gyro_z, &buffer[22], 4);
    memcpy(&data.motor_angle_L, &buffer[26], 4);
    memcpy(&data.motor_angle_R, &buffer[30], 4);
    memcpy(&data.motor_vel_L, &buffer[34], 4);
    memcpy(&data.motor_vel_R, &buffer[38], 4);
    data.pattern_result = buffer[42];
    memcpy(&data.torque, &buffer[43], 4);
    memcpy(&data.Confidence, &buffer[47], 4);
    return data;
}

bool first_in = true;
bool find_new_head = false;
uint8_t buffer[256];
std::deque<uint8_t> buff_dy;

// 核心处理线程
void processSensor() {
    while (!flg_exit.load()) {
        memset(buffer, 0, sizeof(buffer));
        fd_set readfds;
        struct timeval tv;
        
        FD_ZERO(&readfds);
        FD_SET(serial_port, &readfds);
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms 超时
        
        int ret = select(serial_port + 1, &readfds, NULL, NULL, &tv);
        if (ret < 0 || flg_exit.load()) break;
        if (ret == 0) continue;
        
        int bytes_read = read(serial_port, buffer, sizeof(buffer) - 1);
        if (bytes_read <= 0 || bytes_read > 100) continue;
        
        if (first_in) {
            first_in = false;
            bool find_head = false;
            for (int data_i = 0; data_i < bytes_read; data_i++) {
                if (buffer[data_i] == SENSOR_HEAD1 && (data_i == bytes_read-1 || buffer[data_i+1] == SENSOR_HEAD2)) {
                    find_head = true; break;
                }
            }
            if (find_head) {
                display_message("Found sensor header detected, starting sync...");
                bytes_read = 0; 
            }
        }

        if (bytes_read > 0) {
            for (int i = 0; i < bytes_read; ++i) {
                buff_dy.push_back(buffer[i]);
            }
        }

        while (buff_dy.size() >= SENSOR_FRAME_SIZE) { 
            if (find_new_head) {
                find_new_head = false;
                while (!buff_dy.empty()) {
                    uint8_t h1 = buff_dy.front();
                    bool valid_head = false;
                    if (buff_dy.size() > 1) {
                        if (h1 == SENSOR_HEAD1 && buff_dy[1] == SENSOR_HEAD2) valid_head = true;
                    }
                    if (valid_head) break;
                    buff_dy.pop_front();
                }
                if (buff_dy.size() < SENSOR_FRAME_SIZE) break;
            }

            uint8_t temp_buffer[SENSOR_FRAME_SIZE];
            for (int i = 0; i < SENSOR_FRAME_SIZE; ++i) {
                temp_buffer[i] = buff_dy.front();
                buff_dy.pop_front();
            }

            if (temp_buffer[0] == SENSOR_HEAD1 && temp_buffer[1] == SENSOR_HEAD2) {
                if (temp_buffer[51] == SENSOR_END1 && temp_buffer[52] == SENSOR_END2) {
                    SensorData data = parseSensorData(temp_buffer);
                    
                    // ==========================================
                    // === 新增：电机零偏校准与波动检测逻辑 ===
                    // ==========================================
                    if (!is_calibrated.load()) {
                        calib_buf_L.push_back(data.motor_angle_L);
                        calib_buf_R.push_back(data.motor_angle_R);
                        
                        if (calib_buf_L.size() >= CALIB_SAMPLES) {
                            // 计算波动范围 (Max - Min)
                            float min_L = *std::min_element(calib_buf_L.begin(), calib_buf_L.end());
                            float max_L = *std::max_element(calib_buf_L.begin(), calib_buf_L.end());
                            float min_R = *std::min_element(calib_buf_R.begin(), calib_buf_R.end());
                            float max_R = *std::max_element(calib_buf_R.begin(), calib_buf_R.end());
                            
                            if ((max_L - min_L) < STABILITY_THRESHOLD && (max_R - min_R) < STABILITY_THRESHOLD) {
                                // 波动足够小，计算平均值作为偏置
                                float sum_L = std::accumulate(calib_buf_L.begin(), calib_buf_L.end(), 0.0f);
                                float sum_R = std::accumulate(calib_buf_R.begin(), calib_buf_R.end(), 0.0f);
                                offset_motor_L = sum_L / CALIB_SAMPLES;
                                offset_motor_R = sum_R / CALIB_SAMPLES;
                                
                                is_calibrated.store(true);
                                std::cout << "\n>>> [Calibration] Success! Standing still detected." << std::endl;
                                std::cout << ">>> Offset L: " << offset_motor_L << ", Offset R: " << offset_motor_R << "\n" << std::endl;
                            } else {
                                // 波动过大，说明未站稳，清空缓冲区重新收集
                                calib_buf_L.clear();
                                calib_buf_R.clear();
                                
                                static int wait_cnt = 0;
                                if (wait_cnt++ % 5 == 0) { 
                                    display_message("High fluctuation detected. Waiting for standing still...");
                                }
                            }
                        }
                        // 校准未完成前，跳过后面的存储与记录流程
                        continue; 
                    }

                    // 已经校准完成，减去偏置值
                    data.motor_angle_L -= offset_motor_L;
                    data.motor_angle_R -= offset_motor_R;
                    // ==========================================


                    {
                        std::lock_guard<std::mutex> lock(data_mutex);
                        global_data_buffer[0] = data.acc_x;     global_data_buffer[1] = data.acc_y;
                        global_data_buffer[2] = data.acc_z;     global_data_buffer[3] = data.gyro_x;
                        global_data_buffer[4] = data.gyro_y;    global_data_buffer[5] = data.gyro_z;
                        global_data_buffer[6] = data.motor_angle_L; global_data_buffer[7] = data.motor_angle_R;
                        global_data_buffer[8] = data.motor_vel_L;   global_data_buffer[9] = data.motor_vel_R;
                        global_data_buffer[10] = static_cast<float>(data.pattern_result);
                        global_data_buffer[11] = data.torque;
                        global_data_buffer[12] = data.Confidence;
                    }

                    {
                        std::lock_guard<std::mutex> lock(history_mutex);
                        DataRecord record;
                        record.acc[0] = data.acc_x;     record.acc[1] = data.acc_y;     record.acc[2] = data.acc_z;
                        record.gyro[0] = data.gyro_x;   record.gyro[1] = data.gyro_y;   record.gyro[2] = data.gyro_z;
                        record.motor_ang[0] = data.motor_angle_L; record.motor_ang[1] = data.motor_angle_R;
                        record.motor_vel[0] = data.motor_vel_L;   record.motor_vel[1] = data.motor_vel_R;
                        record.pattern_result = data.pattern_result;
                        record.torque = data.torque;
                        record.confidence = data.Confidence;
                        record.timestamp = get_time_sec();
                        
                        data_history.push_back(record);
                        while (data_history.size() > MAX_HISTORY) data_history.pop_front();
                    }

                    double timeNow = get_time_sec();
                    const int current_flag = flag_num.load();
                    if (current_flag == 2) {
                        total_flag2_samples.fetch_add(1);
                        if (data.pattern_result == 2) {
                            total_pattern2_samples.fetch_add(1);
                        }
                    }

                    dataStream.open(file_path_data, std::ios::app);
                    dataStream << std::fixed << std::setprecision(5) << timeNow << ","
                             << data.acc_x << "," << data.acc_y << "," << data.acc_z << ","
                             << data.gyro_x << "," << data.gyro_y << "," << data.gyro_z << ","
                             << data.motor_angle_L << "," << data.motor_angle_R << ","
                             << data.motor_vel_L << "," << data.motor_vel_R << ","
                             << static_cast<int>(data.pattern_result) << ","
                             << data.torque << ","
                             << data.Confidence << ","
                             << current_flag << std::endl;
                    dataStream.close();

                    if (sensor_debug_out) {
                        display_message("[Sensor] Debug trigger");
                        sensor_debug_out = false;
                    }
                } else {
                    for (int i = SENSOR_FRAME_SIZE - 1; i > 1; i--) buff_dy.push_front(temp_buffer[i]);
                    find_new_head = true; continue;
                }
            } else {
                display_message("Data corrupted, resyncing...");
                find_new_head = true;
            }
        }
    }
}

// 初始化CSV表头
void init_csv(std::string full_path) {
    std::fstream fs;
    fs.open(full_path, std::ios::out);
    if (!fs.is_open()) {
        display_message("Cannot create file: " + full_path);
        return;
    }
    fs << "timeStamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,"
       << "motor_angle_L,motor_angle_R,motor_vel_L,motor_vel_R,pattern_result,torque,confidence,flag_num,"
       << "flag2_total_samples,flag2_pattern2_samples,flag2_ratio" << std::endl;
    fs.close();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <serial_port>" << std::endl;
        return 1;
    }

    const char *uartName = argv[1];
    signal(SIGINT, SigHandle);   
    signal(SIGTSTP, SigHandle);  

    // === 配置标准终端输入为非阻塞、无回显模式 ===
    struct termios orig_termios, new_termios;
    int oldf = 0;
    if (keyBoardCatch) {
        tcgetattr(STDIN_FILENO, &orig_termios); 
        new_termios = orig_termios;
        new_termios.c_lflag &= ~(ICANON | ECHO); 
        tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
        
        oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    }
    // ============================================

    std::string time_str = get_current_datetime_str();      
    file_path_data = base_path + "sensor_motor_data_" + time_str + ".csv";
    std::cout << "Saving all data to: " << file_path_data << std::endl;
    init_csv(file_path_data);

    // Serial Setup
    serial_port = open(uartName, O_RDWR);
    if (serial_port < 0) {
        if (keyBoardCatch) {
            tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
            fcntl(STDIN_FILENO, F_SETFL, oldf);
        }
        std::cerr << "Error opening serial port" << std::endl;
        return 1;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(serial_port, &tty) != 0) {
        std::cerr << "Error getting serial attributes" << std::endl;
        close(serial_port);
        if (keyBoardCatch) {
            tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
            fcntl(STDIN_FILENO, F_SETFL, oldf);
        }
        return 1;
    }

    cfsetospeed(&tty, B921600); 
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); 
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~OPOST;

    tcsetattr(serial_port, TCSANOW, &tty);

    // 启动接收线程
    std::thread sensor_thread(processSensor);
    uint16_t debugOutCnt = 0;
    
    std::cout << "\n=========================================" << std::endl;
    std::cout << " System Initialized. Waiting for data..." << std::endl;
    std::cout << " Press 'q' to exit, '0'-'9' to set flag." << std::endl;
    std::cout << "=========================================\n" << std::endl;

    // 主循环
    while (!flg_exit.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // 控制打印显示频率 (10Hz，避免滚屏过快)
        debugOutCnt++;
        if (debugOutCnt >= 10) {  
            debugOutCnt = 0;
            // 只有当校准完成后，才允许在终端输出数据日志
            if (is_calibrated.load()) {
                display_history_status();
            }
        }

        // 捕捉键盘输入 (非阻塞)
        if (keyBoardCatch && !flg_exit.load()) {
            int ch_ = getchar();
            if (ch_ != EOF) {
                if (ch_ >= '0' && ch_ <= '9') {
                    flag_num.store(ch_ - '0');
                    std::cout << "\n>>> [User Input] Flag set to: " << flag_num.load() << " <<<\n" << std::endl;
                } 
                else if (ch_ == 'q' || ch_ == 'Q') {
                    flg_exit.store(true);
                    std::cout << "\n>>> Quit key pressed, preparing to exit... <<<\n" << std::endl;
                }
            }
        }
    }

    // 退出清理 - 只执行一次
    if (!cleanup_done.exchange(true)) {
        std::cout << "Cleaning up resources..." << std::endl;
        
        // 恢复原始终端设置
        if (keyBoardCatch) {
            tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
            fcntl(STDIN_FILENO, F_SETFL, oldf);
        }
        
        close(serial_port);
        
        if (sensor_thread.joinable()) {
            sensor_thread.join();
        } 

        const uint64_t flag2_total = total_flag2_samples.load();
        const uint64_t pattern2_total = total_pattern2_samples.load();
        const double flag2_ratio =
            (flag2_total > 0) ? static_cast<double>(pattern2_total) / static_cast<double>(flag2_total) : 0.0;

        dataStream.open(file_path_data, std::ios::app);
        if (dataStream.is_open()) {
            dataStream << ",,,,,,,,,,,,,,,"
                       << flag2_total << ","
                       << pattern2_total << ","
                       << std::fixed << std::setprecision(6) << flag2_ratio << std::endl;
            dataStream.close();
        }

        std::cout << "flag=2 total samples: " << flag2_total << std::endl;
        std::cout << "pattern=2 samples during flag=2: " << pattern2_total << std::endl;
        std::cout << "ratio: " << std::fixed << std::setprecision(6) << flag2_ratio << std::endl;
        std::cout << "Program finished successfully." << std::endl;
    }
    
    return 0;
}