#ifndef _UART_SERIAL_H
#define _UART_SERIAL_H

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <memory.h>
#include <memory>
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
    explicit SerialBase(int fd);

public:
    static std::unique_ptr<SerialBase> create(const std::string& device,
                                              speed_t speed = B115200,
                                              int file_flag = O_RDWR | O_NOCTTY);

    virtual bool init(speed_t speed = B115200);
    virtual ~SerialBase();

    int getFd() const;
    bool isValid() const;
    int send(const std::string& msg);
    int send(const void* buf, int n_bytes);
    int receive(void *buf, int n_bytes);

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

    inline bool matchHead();

    inline bool matchEnd(const uint8_t* frame);

    void findNextHead();

    void processLoop();

protected:
    explicit SerialLoop(int fd,
                        const std::vector<uint8_t>& data_head,
                        const std::vector<uint8_t>& data_end,
                        const timeval& timeout);

public:
    static std::unique_ptr<SerialLoop<FRAME_LEN>> create(
                        const std::string& device,
                        const std::vector<uint8_t>& data_head,
                        const std::vector<uint8_t>& data_end,
                        speed_t speed = B115200,
                        const timeval& timeout = {0, 10000},
                        int file_flag = O_RDWR | O_NOCTTY);

    ~SerialLoop();
    bool init(speed_t speed = B115200) override;
    void setDataFormat(const std::vector<uint8_t>& data_head,
                       const std::vector<uint8_t>& data_end);
    void setCallback(DataCallback cb);
    void setTimeout(int sec, int usec);
    void start();
    void stop();
    bool isRunning() const;
    void clearBuffer();
    size_t getBufferSize() const;
    uint32_t swapU32(uint32_t val);
};


#endif