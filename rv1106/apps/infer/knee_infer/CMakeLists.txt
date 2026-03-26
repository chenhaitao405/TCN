# ==================== Knee Pattern 推理程序 ====================
add_executable(knee_pattern_infer knee_pattern_infer.cpp)
target_include_directories(knee_pattern_infer PRIVATE
    ${RKNN_API_PATH}/include
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(knee_pattern_infer
    knee_data_producer
    dsp_filters
    timing_utils
    ${RKNN_RT_LIB}
    Threads::Threads
)

# ==================== Knee Torque 推理程序 ====================
add_executable(knee_torque_infer knee_torque_infer.cpp)
target_include_directories(knee_torque_infer PRIVATE
    ${RKNN_API_PATH}/include
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(knee_torque_infer
    knee_data_producer
    dsp_filters
    timing_utils
    ${RKNN_RT_LIB}
    Threads::Threads
)

# ==================== Knee Torque 推理+录制程序 ====================
add_executable(knee_torque_infer_record knee_torque_infer_record.cpp)
target_include_directories(knee_torque_infer_record PRIVATE
    ${RKNN_API_PATH}/include
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(knee_torque_infer_record
    knee_data_producer
    dsp_filters
    timing_utils
    ${RKNN_RT_LIB}
    Threads::Threads
)

install(TARGETS knee_pattern_infer knee_torque_infer knee_torque_infer_record DESTINATION knee_infer)
