#ifndef _UART_SERIAL_H
#define _UART_SERIAL_H

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <string>
#include <iostream>
#include <memory.h>
#include <atomic>
#include <vector>
#include <array>
#include <thread>
#include <cstring>
#include <functional>
#include <deque>

class SerialBase {
protected:
    int serial_fd_;
    explicit SerialBase(int fd) : serial_fd_(fd) {}

public:
    static std::unique_ptr<SerialBase> create(const std::string& device, 
                                               speed_t speed = B115200,
                                               int file_flag = O_RDWR | O_NOCTTY) {
        int fd = open(device.c_str(), file_flag);
        if (fd == -1) {
            std::cerr << "[Error] 无法打开串口: " << device << " (errno: " << errno << ")" << std::endl;
            return nullptr;
        }
        
        auto serial = std::unique_ptr<SerialBase>(new SerialBase(fd));
        if (!serial->init(speed)) {
            close(fd);
            return nullptr;
        }
        
        return serial;
    }

    virtual bool init(speed_t speed = B115200) {
        if (serial_fd_ == -1) return false;
        
        struct termios options;
        memset(&options, 0, sizeof(options));
        if (tcgetattr(serial_fd_, &options) != 0) {
            std::cerr << "[Error] 获取串口属性失败" << std::endl;
            return false;
        }
        
        cfsetispeed(&options, speed);
        cfsetospeed(&options, speed);
        options.c_cflag = (options.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
        options.c_cflag |= (CLOCAL | CREAD);                    // ignore modem controls
        options.c_cflag &= ~(PARENB | PARODD);                  // no parity
        options.c_cflag &= ~CSTOPB;                             // 1 stop bit
        options.c_cflag &= ~CRTSCTS;                            // no flow control

        options.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
        options.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
        options.c_oflag &= ~OPOST;

        if (tcsetattr(serial_fd_, TCSANOW, &options) != 0) {
            std::cerr << "[Error] 设置串口属性失败" << std::endl;
            return false;
        }
        
        tcflush(serial_fd_, TCIOFLUSH);
        return true;
    }

    virtual ~SerialBase() {
        if (serial_fd_ != -1) {
            close(serial_fd_);
        }
    }

    int getFd() const { return serial_fd_; }

    bool isValid() const { return serial_fd_ != -1; }

    int send(const std::string& msg) {
        if (serial_fd_ != -1) {
            return write(serial_fd_, msg.c_str(), msg.length());
        } else {
            std::cerr << "[Error] 无法打开串口";
            return -1;
        }
    }

    int send(const void* buf, int n_bytes) {
        if (serial_fd_ != -1) {
            return write(serial_fd_, buf, n_bytes);
        } else {
            std::cerr << "[Error] 无法打开串口";
            return -1;
        }
    }

    int receive(void *buf, int n_bytes) {
        if (serial_fd_ != -1) {
            return read(serial_fd_, buf, n_bytes);
        } else {
            std::cerr << "[Error] 无法打开串口";
            return -1;
        }
    }

};

using DataCallback = std::function<void(const uint8_t* frame)>;

template <size_t FRAME_LEN>
class SerialLoop : public SerialBase {
    
private:
    std::atomic<bool> running_;
    std::thread worker_thread_;
    DataCallback callback_;
    uint8_t read_buffer_[256];
    bool find_new_head;
    std::deque<uint8_t> buffer_;
    std::vector<uint8_t> data_head_;
    std::vector<uint8_t> data_end_;
    struct timeval timeout_;

    // 检查缓冲区开头是否匹配帧头
    inline bool matchHead() {
        for (size_t i = 0; i < data_head_.size(); ++i) {
            if (buffer_[i] != data_head_[i]) return false;
        }
        return true;
    }

    // 检查帧尾是否匹配
    inline bool matchEnd(const uint8_t* frame) {
        size_t end_len = data_end_.size();
        for (size_t i = 0; i < end_len; ++i) {
            if (frame[FRAME_LEN - end_len + i] != data_end_[i]) return false;
        }
        return true;
    }

    // 寻找下一个帧头
    void findNextHead() {
        while (buffer_.size() >= data_head_.size()) {
            if (matchHead()) {
                find_new_head = false;
                break;
            }
            buffer_.pop_front();
        }
    }

    void processLoop() {
        find_new_head = true;
        fd_set read_fds;

        while (running_) {
            // 事件驱动：等待串口可读
            FD_ZERO(&read_fds);
            FD_SET(serial_fd_, &read_fds);
            struct timeval timeout = timeout_;
            
            int ready = select(serial_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);

            if (ready <= 0) {
                continue;
            }

            int bytes_read = read(serial_fd_, read_buffer_, sizeof(read_buffer_));
            
            if (bytes_read <= 0) continue;

            // 数据加入缓冲区
            buffer_.insert(buffer_.end(), read_buffer_, read_buffer_ + bytes_read);

            while (buffer_.size() >= FRAME_LEN) {
                if (find_new_head) {
                    findNextHead();
                    // 可能只找到了下一帧的帧头，剩余数据还没传过来
                    if (buffer_.size() < FRAME_LEN) break;
                }

                uint8_t frame[FRAME_LEN];
                for (size_t i = 0; i < FRAME_LEN; ++i) {
                    frame[i] = buffer_[i];
                }

                // 帧头正确则进入
                if (!find_new_head) {
                    if (matchEnd(frame)) {
                        buffer_.erase(buffer_.begin(), buffer_.begin() + FRAME_LEN);
                        find_new_head = true;
                        // 调用回调函数
                        if (callback_) {
                            callback_(frame + data_head_.size());
                        }
                    } else {
                        // 帧尾错误，丢弃头部，寻找下一个头
                        buffer_.pop_front();
                        find_new_head = true;
                    }
                }
            }
        }
    }

protected:
    explicit SerialLoop(int fd,
                        const std::vector<uint8_t>& data_head,
                        const std::vector<uint8_t>& data_end,
                        const timeval& timeout)
        : SerialBase(fd), running_(false), 
          data_head_(data_head), data_end_(data_end), timeout_(timeout) {
    }

public:
    static std::unique_ptr<SerialLoop<FRAME_LEN>> create(
                        const std::string& device,
                        const std::vector<uint8_t>& data_head,
                        const std::vector<uint8_t>& data_end,
                        speed_t speed = B115200,
                        const timeval& timeout = {0, 10000},
                        int file_flag = O_RDWR | O_NOCTTY) {
        
        int fd = open(device.c_str(), file_flag);
        if (fd == -1) {
            std::cerr << "[Error] 无法打开串口: " << device << " (errno: " << errno << ")" << std::endl;
            return nullptr;
        }
        
        auto serial = std::unique_ptr<SerialLoop<FRAME_LEN>>(
            new SerialLoop<FRAME_LEN>(fd, data_head, data_end, timeout));
        
        if (!serial->init(speed)) {
            close(fd);
            return nullptr;
        }
        
        return serial;
    }

    ~SerialLoop() {
        stop();
    }

    bool init(speed_t speed = B115200) override {
        if (!SerialBase::init(speed)) return false;
        if (serial_fd_ == -1) return false;
        buffer_.clear();

        struct termios options;
        if (tcgetattr(serial_fd_, &options) != 0) {
            std::cerr << "[Error] 获取串口属性失败" << std::endl;
            return false;
        }

        // 读阻塞模式
        options.c_cc[VMIN] = 1;
        options.c_cc[VTIME] = 0;

        if (tcsetattr(serial_fd_, TCSANOW, &options) != 0) {
            std::cerr << "[Error] 设置串口属性失败" << std::endl;
            return false;
        }
        
        tcflush(serial_fd_, TCIOFLUSH);
        return true;
    }

    void setDataFormat(const std::vector<uint8_t>& data_head, 
                       const std::vector<uint8_t>& data_end) {
        data_head_ = data_head;
        data_end_ = data_end;
    }

    void setCallback(DataCallback cb) {
        callback_ = cb;
    }

    void setTimeout(int sec, int usec) {
        timeout_.tv_sec = sec;
        timeout_.tv_usec = usec;
    }

    void start() {
        if (running_) return;
        if (serial_fd_ < 0) {
            std::cerr << "[Error] 串口未打开" << std::endl;
            return;
        }
        running_ = true;
        worker_thread_ = std::thread(&SerialLoop::processLoop, this);
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    bool isRunning() const { return running_; }

    // 清空缓冲区
    void clearBuffer() {
        buffer_.clear();
    }

    // 获取缓冲区大小
    size_t getBufferSize() const {
        return buffer_.size();
    }

    uint32_t swapU32(uint32_t val) {
        return ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) | 
               ((val << 8) & 0xFF0000) | ((val << 24) & 0xFF000000);
    }
};


#endif