cmake_minimum_required(VERSION 3.6)

project(gpio_uart)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER /home/sxs/modules/2.3.2/release/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc)
set(CMAKE_CXX_COMPILER /home/sxs/modules/2.3.2/release/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wl,--allow-shlib-undefined -O3")
set(RKNN_API_PATH /home/sxs/modules/2.3.2/release/rknn-toolkit2-v2.3.2-2025-04-09/rknpu2/runtime/Linux/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so)

add_executable(test
               main.cpp)
target_include_directories(test PRIVATE ${RKNN_API_PATH}/include)

target_link_libraries(test ${RKNN_RT_LIB})
