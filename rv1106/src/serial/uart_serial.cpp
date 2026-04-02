#include "serial/uart_serial.h"

#include <cerrno>
#include <sys/select.h>

SerialBase::SerialBase(int fd) : serial_fd_(fd) {}

std::unique_ptr<SerialBase> SerialBase::create(const std::string& device,
                                               speed_t speed,
                                               int file_flag) {
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

bool SerialBase::init(speed_t speed) {
    if (serial_fd_ == -1) return false;

    struct termios options;
    memset(&options, 0, sizeof(options));
    if (tcgetattr(serial_fd_, &options) != 0) {
        std::cerr << "[Error] 获取串口属性失败" << std::endl;
        return false;
    }

    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);
    options.c_cflag = (options.c_cflag & ~CSIZE) | CS8;
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~(PARENB | PARODD);
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CRTSCTS;

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

SerialBase::~SerialBase() {
    if (serial_fd_ != -1) {
        close(serial_fd_);
    }
}

int SerialBase::getFd() const { return serial_fd_; }

bool SerialBase::isValid() const { return serial_fd_ != -1; }

int SerialBase::send(const std::string& msg) {
    if (serial_fd_ != -1) {
        return write(serial_fd_, msg.c_str(), msg.length());
    }
    std::cerr << "[Error] 无法打开串口";
    return -1;
}

int SerialBase::send(const void* buf, int n_bytes) {
    if (serial_fd_ != -1) {
        return write(serial_fd_, buf, n_bytes);
    }
    std::cerr << "[Error] 无法打开串口";
    return -1;
}

int SerialBase::receive(void *buf, int n_bytes) {
    if (serial_fd_ != -1) {
        return read(serial_fd_, buf, n_bytes);
    }
    std::cerr << "[Error] 无法打开串口";
    return -1;
}

template <size_t FRAME_LEN>
inline bool SerialLoop<FRAME_LEN>::matchHead() {
    for (size_t i = 0; i < data_head_.size(); ++i) {
        if (buffer_[i] != data_head_[i]) return false;
    }
    return true;
}

template <size_t FRAME_LEN>
inline bool SerialLoop<FRAME_LEN>::matchEnd(const uint8_t* frame) {
    size_t end_len = data_end_.size();
    for (size_t i = 0; i < end_len; ++i) {
        if (frame[FRAME_LEN - end_len + i] != data_end_[i]) return false;
    }
    return true;
}

template <size_t FRAME_LEN>
void SerialLoop<FRAME_LEN>::findNextHead() {
    while (buffer_.size() >= data_head_.size()) {
        if (matchHead()) {
            find_new_head = false;
            break;
        }
        buffer_.pop_front();
    }
}

template <size_t FRAME_LEN>
void SerialLoop<FRAME_LEN>::processLoop() {
    find_new_head = true;
    fd_set read_fds;

    while (running_) {
        FD_ZERO(&read_fds);
        FD_SET(serial_fd_, &read_fds);
        struct timeval timeout = timeout_;

        int ready = select(serial_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);

        if (ready <= 0) {
            continue;
        }

        int bytes_read = read(serial_fd_, read_buffer_, sizeof(read_buffer_));
        if (bytes_read <= 0) continue;

        buffer_.insert(buffer_.end(), read_buffer_, read_buffer_ + bytes_read);

        while (buffer_.size() >= FRAME_LEN) {
            if (find_new_head) {
                findNextHead();
                if (buffer_.size() < FRAME_LEN) break;
            }

            uint8_t frame[FRAME_LEN];
            for (size_t i = 0; i < FRAME_LEN; ++i) {
                frame[i] = buffer_[i];
            }

            if (!find_new_head) {
                if (matchEnd(frame)) {
                    buffer_.erase(buffer_.begin(), buffer_.begin() + FRAME_LEN);
                    find_new_head = true;
                    if (callback_) {
                        callback_(frame + data_head_.size());
                    }
                } else {
                    buffer_.pop_front();
                    find_new_head = true;
                }
            }
        }
    }
}

template <size_t FRAME_LEN>
SerialLoop<FRAME_LEN>::SerialLoop(int fd,
                                  const std::vector<uint8_t>& data_head,
                                  const std::vector<uint8_t>& data_end,
                                  const timeval& timeout)
    : SerialBase(fd), running_(false), data_head_(data_head), data_end_(data_end), timeout_(timeout) {}

template <size_t FRAME_LEN>
std::unique_ptr<SerialLoop<FRAME_LEN>> SerialLoop<FRAME_LEN>::create(
    const std::string& device,
    const std::vector<uint8_t>& data_head,
    const std::vector<uint8_t>& data_end,
    speed_t speed,
    const timeval& timeout,
    int file_flag) {
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

template <size_t FRAME_LEN>
SerialLoop<FRAME_LEN>::~SerialLoop() {
    stop();
}

template <size_t FRAME_LEN>
bool SerialLoop<FRAME_LEN>::init(speed_t speed) {
    if (!SerialBase::init(speed)) return false;
    if (serial_fd_ == -1) return false;
    buffer_.clear();

    struct termios options;
    if (tcgetattr(serial_fd_, &options) != 0) {
        std::cerr << "[Error] 获取串口属性失败" << std::endl;
        return false;
    }

    options.c_cc[VMIN] = 1;
    options.c_cc[VTIME] = 0;

    if (tcsetattr(serial_fd_, TCSANOW, &options) != 0) {
        std::cerr << "[Error] 设置串口属性失败" << std::endl;
        return false;
    }

    tcflush(serial_fd_, TCIOFLUSH);
    return true;
}

template <size_t FRAME_LEN>
void SerialLoop<FRAME_LEN>::setDataFormat(const std::vector<uint8_t>& data_head,
                                          const std::vector<uint8_t>& data_end) {
    data_head_ = data_head;
    data_end_ = data_end;
}

template <size_t FRAME_LEN>
void SerialLoop<FRAME_LEN>::setCallback(DataCallback cb) {
    callback_ = cb;
}

template <size_t FRAME_LEN>
void SerialLoop<FRAME_LEN>::setTimeout(int sec, int usec) {
    timeout_.tv_sec = sec;
    timeout_.tv_usec = usec;
}

template <size_t FRAME_LEN>
void SerialLoop<FRAME_LEN>::start() {
    if (running_) return;
    if (serial_fd_ < 0) {
        std::cerr << "[Error] 串口未打开" << std::endl;
        return;
    }
    running_ = true;
    worker_thread_ = std::thread(&SerialLoop::processLoop, this);
}

template <size_t FRAME_LEN>
void SerialLoop<FRAME_LEN>::stop() {
    if (!running_) return;
    running_ = false;
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

template <size_t FRAME_LEN>
bool SerialLoop<FRAME_LEN>::isRunning() const {
    return running_;
}

template <size_t FRAME_LEN>
void SerialLoop<FRAME_LEN>::clearBuffer() {
    buffer_.clear();
}

template <size_t FRAME_LEN>
size_t SerialLoop<FRAME_LEN>::getBufferSize() const {
    return buffer_.size();
}

template <size_t FRAME_LEN>
uint32_t SerialLoop<FRAME_LEN>::swapU32(uint32_t val) {
    return ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
           ((val << 8) & 0xFF0000) | ((val << 24) & 0xFF000000);
}

template class SerialLoop<20>;
template class SerialLoop<48>;
