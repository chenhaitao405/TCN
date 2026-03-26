# ==================== 测试程序 ====================

# --- 功耗温度测试（需要 RKNN） ---
add_executable(power_temp_test power_temp_test.cpp)
target_include_directories(power_temp_test PRIVATE
    ${RKNN_API_PATH}/include
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(power_temp_test
    ${RKNN_RT_LIB}
)

# --- CPU+NPU 满载功耗测试（需要 RKNN） ---
add_executable(full_power_test full_power_test.cpp)
target_include_directories(full_power_test PRIVATE
    ${RKNN_API_PATH}/include
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(full_power_test
    ${RKNN_RT_LIB}
    Threads::Threads
)

# --- 预处理速度对比（需要 RKNN） ---
add_executable(preprocess_speed_compare preprocess_speed_compare.cpp)
target_include_directories(preprocess_speed_compare PRIVATE
    ${RKNN_API_PATH}/include
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(preprocess_speed_compare
    ${RKNN_RT_LIB}
)

# --- 预处理结果一致性对比 ---
add_executable(preprocess_result_test preprocess_result_test.cpp)
target_include_directories(preprocess_result_test PRIVATE
    ${RKNN_API_PATH}/include
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(preprocess_result_test
    ${RKNN_RT_LIB}
)

# --- GPIO + UART 测试 ---
add_executable(test_gpio_uart test_gpio_uart.cpp)
target_include_directories(test_gpio_uart PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/serial
    ${PROJECT_INCLUDE_DIR}/gpio
)
target_link_libraries(test_gpio_uart
    Threads::Threads
)

# --- IMU 测试 ---
add_executable(test_imu test_imu.cpp)
target_include_directories(test_imu PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/peripherals
    ${PROJECT_INCLUDE_DIR}/gpio
)
target_link_libraries(test_imu
    peripherals
    Threads::Threads
)

# --- QMC5883P 磁力计测试 ---
add_executable(test_qmc5883p test_qmc5883p.cpp)
target_include_directories(test_qmc5883p PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/peripherals
)
target_link_libraries(test_qmc5883p
    peripherals
)

# --- BM85163 RTC 测试 ---
add_executable(test_bm85163 test_bm85163.cpp)
target_include_directories(test_bm85163 PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/peripherals
)
target_link_libraries(test_bm85163
    peripherals
)

install(TARGETS
    power_temp_test full_power_test preprocess_speed_compare preprocess_result_test
    test_gpio_uart test_imu
    test_qmc5883p test_bm85163
    DESTINATION test
)
