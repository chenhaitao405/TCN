#include <stdio.h>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <signal.h>

#include "gpio.h"
#include "uart_serial.h"

static std::atomic<bool> g_running{true};

void send_1(SerialPort *sp) {
    while (g_running) {
        sp->send("+1\r\n");
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    printf("[Thread] UART thread exit\n");
}

static void signal_handler(int signum) {
    printf("\nReceived signal %d, stopping...\n", signum);
    g_running = false;
}

void invert_level(GpioPin *gp) {
    int state = 0;
    while (g_running) {
        state = !state;
        gp->write(state);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    printf("[Thread] GPIO thread exit\n");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf( "Missing params!\n"
                "Usage: %s serial_port gpio_id\n",argv[0]);
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    SerialPort uart_port(argv[1], B115200);
    GpioPin gpio(atoi(argv[2]), true);

    std::thread uart_thread(send_1, &uart_port);
    std::thread gpio_thread(invert_level, &gpio);

    if (uart_thread.joinable()) uart_thread.join();
    if (gpio_thread.joinable()) gpio_thread.join();

    printf("All Tasks done!\n");
    return 0;
}

