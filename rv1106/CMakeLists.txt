cmake_minimum_required(VERSION 3.6)

# ==================== 交叉编译工具链 ====================
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(TOOLCHAIN_ROOT /home/sxs/modules/2.3.2/release/arm-rockchip830-linux-uclibcgnueabihf)
set(CMAKE_C_COMPILER   ${TOOLCHAIN_ROOT}/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_ROOT}/bin/arm-rockchip830-linux-uclibcgnueabihf-g++)

project(rv1106_apps CXX C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==================== 公共编译选项 ====================
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--allow-shlib-undefined")

find_package(Threads REQUIRED)

# ==================== 安装路径 ====================
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${CMAKE_SYSTEM_NAME}/apps)
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# ==================== RKNN SDK ====================
set(RKNN_API_PATH /home/sxs/modules/2.3.2/release/rknn-toolkit2-v2.3.2-2025-04-09/rknpu2/runtime/Linux/librknn_api)
set(RKNN_RT_LIB   ${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so)

set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# ==================== 功能开关 ====================
option(ENABLE_DEBUG        "Enable debug print"          ON)
option(ENABLE_FALL_DETECT  "Enable fall detection"       OFF)
option(ENABLE_FILTER_INPUT "Enable motor velocity filter" ON)
option(ENABLE_HIP_MODEL_10_INPUT_CHANNEL "Hip model 10 input channels, otherwise 8 input channels" OFF)

if(ENABLE_DEBUG)
    add_compile_definitions(DEBUG)
endif()
if(ENABLE_FALL_DETECT)
    add_compile_definitions(FAIL_DETECT)
endif()
if(ENABLE_FILTER_INPUT)
    add_compile_definitions(FILTER_INPUT)
endif()
if(ENABLE_HIP_MODEL_10_INPUT_CHANNEL)
    add_compile_definitions(HIP_MODEL_10_INPUT_CHANNEL)
endif()

add_subdirectory(src)

add_subdirectory(apps)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(FILES ../deploy/trained_quantcn_8_sensors.rknn DESTINATION ./model)
install(FILES ../timesnet_rknn/output/modes_cnn_int8.rknn DESTINATION ./model)
