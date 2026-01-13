cmake_minimum_required(VERSION 3.6)

project(tcn)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER /home/sxs/modules/2.3.2/release/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc)
set(CMAKE_CXX_COMPILER /home/sxs/modules/2.3.2/release/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/npy_infer_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
set(RKNN_API_PATH /home/sxs/modules/2.3.2/release/rknn-toolkit2-v2.3.2-2025-04-09/rknpu2/runtime/Linux/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so)

include_directories(${RKNN_API_PATH}/include)

add_executable(npy_infer
               npy_infer.cpp)

target_link_libraries(npy_infer
    ${RKNN_RT_LIB}
)

# install target and libraries
install(TARGETS npy_infer DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(FILES ../../deploy/trained_quantcn_8_sensors.rknn DESTINATION ./model)

install(FILES ../../deploy/datasets/jump_1.npy DESTINATION ./model/)
install(FILES ../../deploy/datasets/jump_10.npy DESTINATION ./model/)