# ==================== Hip 推理程序 ====================
add_executable(hip_pattern_infer hip_pattern_infer.cpp)
target_include_directories(hip_pattern_infer PRIVATE
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(hip_pattern_infer
    hip_data_producer
    dsp_filters
    timing_utils
    ${RKNN_RT_LIB}
    Threads::Threads
)

add_executable(hip_torque_infer hip_torque_infer.cpp)
target_include_directories(hip_torque_infer PRIVATE
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(hip_torque_infer
    hip_data_producer
    dsp_filters
    timing_utils
    ${RKNN_RT_LIB}
    Threads::Threads
)

# ==================== Hip 推理+录制程序 ====================
add_executable(hip_torque_infer_record hip_torque_infer_record.cpp)
target_include_directories(hip_torque_infer_record PRIVATE
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(hip_torque_infer_record
    hip_data_producer
    dsp_filters
    timing_utils
    ${RKNN_RT_LIB}
    Threads::Threads
)

install(TARGETS hip_pattern_infer hip_torque_infer hip_torque_infer_record DESTINATION hip_infer)
