#include "serial/uart_serial.h"
#include "peripherals/nt26-kcn.h"
#include "gpio/gpio.h"
#include <memory>
#include <stdio.h>
#include "peripherals/lsm6dsr.h"
#include <atomic>
#include <vector>
#include <sys/epoll.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <chrono>
#include <cctype>

const int BATCH_SIZE = 10;

static std::unique_ptr<NT26KCN> nt26;
std::atomic<bool> running(true);

bool wait_for_ok(NT26KCN* serial, int timeout_ms) {
    if (!serial || !serial->isValid()) return false;

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::string buffer;
    buffer.reserve(256);

    auto is_token = [](const std::string& buf, size_t pos, const std::string& token) {
        size_t end = pos + token.size();
        if (end > buf.size()) return false;
        if (buf.compare(pos, token.size(), token) != 0) return false;
        bool left_ok = (pos == 0) || !std::isalnum(static_cast<unsigned char>(buf[pos - 1]));
        bool right_ok = (end == buf.size()) || !std::isalnum(static_cast<unsigned char>(buf[end]));
        return left_ok && right_ok;
    };

    int epfd = epoll_create1(0);
    if (epfd < 0) return false;

    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = serial->getFd();
    epoll_ctl(epfd, EPOLL_CTL_ADD, serial->getFd(), &ev);

    while (std::chrono::steady_clock::now() < deadline) {
        auto now = std::chrono::steady_clock::now();
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        if (remaining.count() < 0) break;

        struct epoll_event events[1];
        int ready = epoll_wait(epfd, events, 1, static_cast<int>(remaining.count()));
        if (ready <= 0) continue;

        char temp[256];
        int n = serial->receive(temp, sizeof(temp));
        if (n <= 0) continue;

        buffer.append(temp, temp + n);

        // 先快速检测无换行场景
        size_t ok_pos = buffer.find("OK");
        if (ok_pos != std::string::npos && is_token(buffer, ok_pos, "OK")) { close(epfd); return true; }
        size_t err_pos = buffer.find("ERROR");
        if (err_pos != std::string::npos && is_token(buffer, err_pos, "ERROR")) { close(epfd); return false; }

        // 兼容 \r 或 \n 分隔
        size_t pos = 0;
        while ((pos = buffer.find_first_of("\r\n")) != std::string::npos) {
            std::string line = buffer.substr(0, pos);
            buffer.erase(0, pos + 1);

            if (line == "OK") { close(epfd); return true; }
            if (line == "ERROR") { close(epfd); return false; }
        }

        // 防止无限增长
        if (buffer.size() > 1024) {
            buffer.erase(0, buffer.size() - 256);
        }
    }

    close(epfd);
    return false;
}

bool send_at_wait_ok(NT26KCN* serial, const std::string& cmd, int timeout_ms) {
    if (!serial || !serial->isValid()) return false;
    // 发送前先清空残留数据，避免把上电/上一条指令的 OK 当成当前响应
    auto drain = [](NT26KCN* s, int max_ms) {
        if (!s || !s->isValid()) return;
        int epfd = epoll_create1(0);
        if (epfd < 0) return;
        struct epoll_event ev;
        ev.events = EPOLLIN;
        ev.data.fd = s->getFd();
        epoll_ctl(epfd, EPOLL_CTL_ADD, s->getFd(), &ev);
        auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_ms);
        while (std::chrono::steady_clock::now() < end) {
            struct epoll_event events[1];
            int ready = epoll_wait(epfd, events, 1, 0);
            if (ready <= 0) break;
            char tmp[256];
            int n = s->receive(tmp, sizeof(tmp));
            if (n <= 0) break;
        }
        close(epfd);
    };

    drain(serial, 2);
    int sent = serial->send(cmd.c_str());
    if (sent < 0) return false;
    return wait_for_ok(serial, timeout_ms);
}

void send_batch_data(NT26KCN* serial, const std::vector<ImuMotorData>& batch) {
    if (batch.empty()) return;
    
    // 格式：[count(2B)][ImuMotorData1][ImuMotorData2][ImuMotorData3...]...
    size_t raw_size = 2 + batch.size() * sizeof(ImuMotorData);
    std::vector<uint8_t> payload;
    payload.reserve(raw_size);

    uint16_t count = static_cast<uint16_t>(batch.size());

    // 填充数据
    uint8_t* p_count = reinterpret_cast<uint8_t*>(&count);
    payload.insert(payload.end(), p_count, p_count + 2);

    const uint8_t* p_data = reinterpret_cast<const uint8_t*>(batch.data());
    payload.insert(payload.end(), p_data, p_data + (batch.size() * sizeof(ImuMotorData)));

    // 发送指令
    // 注意：QMTPUBEX 的长度参数是实际要发的字节数
    
    std::string cmd = "AT+QMTPUBEX=0,1,0,0,/zyy/imu_data," + std::to_string(payload.size()) + "\r";
    send_at_wait_ok(serial, cmd, 10);
    // serial->send(cmd.c_str());
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    serial->send(payload.data(), payload.size());
}

void get_imu_data(Lsm6dsr *sensor, GpioPin *gpio, NT26KCN* serial) {
    std::string value_path = gpio->get_value_path();
    int gpio_fd = open(value_path.c_str(), O_RDONLY | O_NONBLOCK);
    
    if (gpio_fd < 0) {
        std::cerr << "[Error] Failed to open GPIO value fd: " << value_path << std::endl;
        return;
    }
    if (sensor->is_debug()) std::cout << "[Debug] GPIO fd opened: " << value_path << std::endl;

    std::vector<ImuMotorData> batch_buffer;
    batch_buffer.reserve(BATCH_SIZE);
    // Setup epoll BEFORE restarting sensor
    int epfd = epoll_create1(0);
    if (epfd < 0) {
        std::cerr << "[Error] Failed to create epoll fd: " << strerror(errno) << std::endl;
        close(gpio_fd);
        return;
    }
    struct epoll_event ev;
    ev.events = EPOLLPRI | EPOLLERR;
    ev.data.fd = gpio_fd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, gpio_fd, &ev);

    // Clear any pending events
    char buf[8];
    lseek(gpio_fd, 0, SEEK_SET);
    read(gpio_fd, buf, sizeof(buf));

    int interrupt_count = 0;
    int poll_timeout_count = 0;

    sensor->startFifo();
    while (running) {
        
        // Wait for interrupt (timeout 20ms)
        struct epoll_event events[1];
        int ret = epoll_wait(epfd, events, 1, 20);

        if (ret > 0) {
            if (events[0].events & EPOLLPRI) {
                // Interrupt occurred: Clear file pointer
                lseek(gpio_fd, 0, SEEK_SET);
                read(gpio_fd, buf, sizeof(buf));

                interrupt_count++;
                
                // Read ALL data from FIFO to let INT1 go LOW
                std::vector<ImuData> data = sensor->readFifo();
                
                // Process Data
                for (const auto& d : data) {
                    ImuMotorData item;
                    item.imu_data = d;
                    item.motor_data = MotorData{};
                    batch_buffer.emplace_back(item);
                }
                if (batch_buffer.size() >= BATCH_SIZE) {
                    send_batch_data(serial, batch_buffer);
                    batch_buffer.clear();
                }
            }
        } else if (ret == 0) {
            // Timeout - print debug info periodically
            poll_timeout_count++;
            lseek(gpio_fd, 0, SEEK_SET);
            int n = read(gpio_fd, buf, sizeof(buf));
            if (n > 0 && buf[0] == '1') {
                if (sensor->is_debug()) {
                    std::cout << "[Debug] GPIO stuck HIGH. Force reading FIFO... ";
                }
                // clean FIFO
                std::vector<ImuData> data = sensor->readFifo();
                if (sensor->is_debug()) std::cout << "[Debug] Have read " << data.size() << " samples." << std::endl;
                    
                if (!data.empty()) {
                    if (sensor->is_debug()) std::cout << "[Debug] Sample: XL " << data[0].acc_x << ", " << data[0].acc_y << std::endl;
                } else {
                    if (sensor->is_debug()) std::cout << "[Debug] GPIO is 1 but FIFO Read returned 0 samples!" << std::endl;
                    // 强制重置策略——如果连续读不到，可能需要reset FIFO
                    sensor->stopAndClearFifo();
                    sensor->startFifo();
                }
            }
        } else {
            
            std::cerr << "[Thread] epoll_wait error: " << strerror(errno) << std::endl;
        }
    }
    
    close(epfd);
    close(gpio_fd);
    sensor->stopAndClearFifo();

    if (!batch_buffer.empty()) {
        send_batch_data(serial, batch_buffer);
    }
    
    if (sensor->is_debug()) std::cout << "[Debug] Stopped. Total interrupts: " << interrupt_count << std::endl;
}


int main(int argc, char **argv) {
    
    GpioPin int1(INT1_GPIO_PIN, GpioPin::DIRECTION::IN, GpioPin::EDGE::RISING);
    nt26 = NT26KCN::create(NT26_UART_TTY, B921600);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    GpioPin nt26_en(NT26_PWRON_PIN, GpioPin::DIRECTION::OUT);
    nt26_en.write(1);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    if (!send_at_wait_ok(nt26.get(), "ATE0\r", 15000)) {
        std::cerr << "[Error] ATE0 超时或失败" << std::endl;
        return -1;
    }
    if (!send_at_wait_ok(nt26.get(), "AT+QMTOPEN=0,\"broker.hivemq.com\",1883\r", 15000)) {
        std::cerr << "[Error] QMTOPEN 超时或失败" << std::endl;
        return -1;
    }
    if (!send_at_wait_ok(nt26.get(), "AT+QMTCONN=0\r", 15000)) {
        std::cerr << "[Error] QMTCONN 超时或失败" << std::endl;
        return -1;
    }
    auto lsm = Lsm6dsr::create(I2C_BUS_ID, LSM_ADDR,
                               ACC_ODR_104 | ACC_4g,
                               GYRO_ODR_104 | GYRO_250dps,
                               0x44, true, true);
    if (!lsm) {
        std::cerr << "[Error] Lsm6dsr initialization failed!" << std::endl;
        return -1;
    }
    std::thread acquisition_thread(get_imu_data, lsm.get(), &int1, nt26.get());
    std::cout << "[INFO] Press Enter to exit..." << std::endl;
    std::cin.get();

    running = false;
    acquisition_thread.join();
    if (!send_at_wait_ok(nt26.get(), "AT+QMTCLOSE=0\r", 15000)) {
        std::cerr << "[Warn] QMTCLOSE 超时或失败" << std::endl;
    }
    return 0;
}