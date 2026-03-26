#include "timing/periodic_timer.h"

#include <cerrno>
#include <sys/timerfd.h>
#include <unistd.h>

namespace timing {

PeriodicTimer::~PeriodicTimer() {
    stop();
}

bool PeriodicTimer::start(int interval_ms, int first_shot_ms) {
    stop();

    if (interval_ms <= 0) {
        return false;
    }
    if (first_shot_ms < 0) {
        first_shot_ms = interval_ms;
    }

    const int tfd = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
    if (tfd < 0) {
        return false;
    }

    itimerspec its{};
    its.it_value.tv_sec = first_shot_ms / 1000;
    its.it_value.tv_nsec = (first_shot_ms % 1000) * 1000000LL;
    its.it_interval.tv_sec = interval_ms / 1000;
    its.it_interval.tv_nsec = (interval_ms % 1000) * 1000000LL;

    if (timerfd_settime(tfd, 0, &its, nullptr) < 0) {
        close(tfd);
        return false;
    }

    fd_ = tfd;
    return true;
}

bool PeriodicTimer::wait(uint64_t& expirations, const std::atomic<bool>* running) const {
    if (fd_ < 0) {
        return false;
    }

    while (true) {
        if (running && !running->load(std::memory_order_acquire)) {
            return false;
        }

        const ssize_t n = read(fd_, &expirations, sizeof(expirations));
        if (n == static_cast<ssize_t>(sizeof(expirations))) {
            return true;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
}

void PeriodicTimer::stop() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

bool PeriodicTimer::valid() const {
    return fd_ >= 0;
}

} // namespace timing
