#pragma once

#include <atomic>
#include <cstdint>

namespace timing {

class PeriodicTimer {
public:
    PeriodicTimer() = default;
    ~PeriodicTimer();

    PeriodicTimer(const PeriodicTimer&) = delete;
    PeriodicTimer& operator=(const PeriodicTimer&) = delete;

    bool start(int interval_ms, int first_shot_ms = -1);
    bool wait(uint64_t& expirations, const std::atomic<bool>* running = nullptr) const;
    void stop();
    bool valid() const;

private:
    int fd_ = -1;
};

} // namespace timing
