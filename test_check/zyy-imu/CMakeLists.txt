cmake_minimum_required(VERSION 3.10)
project(zyy_imu)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找线程库 (Linux 下使用 std::thread 需要)
find_package(Threads REQUIRED)
find_package(Curses REQUIRED)


# [Eigen] 
include_directories(${PROJECT_SOURCE_DIR}/include/eigen)

# 如果在 Linux 上是通过 apt 安装的 Eigen (sudo apt install libeigen3-dev)
# 请注释掉上面那行，改用下面这两行：
# find_package(Eigen3 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIRS})

# 生成主程序
add_executable(motor_read
    "src/motor_read.cpp" 
)

# 链接库
target_link_libraries(motor_read PRIVATE Threads::Threads ${CURSES_LIBRARIES})