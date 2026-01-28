cmake_minimum_required(VERSION 3.6)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER /home/sxs/modules/2.3.2/release/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc)
set(CMAKE_CXX_COMPILER /home/sxs/modules/2.3.2/release/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++)

project(i2c)

find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wl,--allow-shlib-undefined")

add_executable(test_qmc5883p ${CMAKE_SOURCE_DIR}/test_qmc5883p.cpp ${CMAKE_SOURCE_DIR}/qmc5883p.cpp)

target_include_directories(test_qmc5883p PRIVATE
    ${CMAKE_SOURCE_DIR})

add_executable(test_bm85163 ${CMAKE_SOURCE_DIR}/test_bm85163.cpp ${CMAKE_SOURCE_DIR}/bm85163.cpp)

target_include_directories(test_bm85163 PRIVATE
    ${CMAKE_SOURCE_DIR})

add_executable(test_imu ${CMAKE_SOURCE_DIR}/test_imu.cpp ${CMAKE_SOURCE_DIR}/lsm6dsr.cpp)

target_include_directories(test_imu PRIVATE
    ${CMAKE_SOURCE_DIR}/i2c_imu
    ${CMAKE_SOURCE_DIR}/..)
