# ==================== 数据采集程序 ====================

# --- 串口数据采集：RF 数据采集 ---
add_executable(rf_data_collection_hip by_serial/rf_data_collection_hip.cpp)
target_include_directories(rf_data_collection_hip PRIVATE
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(rf_data_collection_hip
    peripherals
    Threads::Threads
    ${RKNN_RT_LIB}
)

add_executable(rf_data_collection_knee by_serial/rf_data_collection_knee.cpp)
target_include_directories(rf_data_collection_knee PRIVATE
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(rf_data_collection_knee
    peripherals
    Threads::Threads
    ${RKNN_RT_LIB}
)

# --- 串口数据采集：电机数据读取 ---
# 注意：motor_read 依赖 ncurses，仅在原生编译环境下可用
# add_executable(motor_read by_serial/motor_read.cpp)
# target_link_libraries(motor_read Threads::Threads ncurses)

install(TARGETS rf_data_collection_hip rf_data_collection_knee DESTINATION data_collection)
