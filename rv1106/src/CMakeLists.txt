# ==================== 外设驱动库 ====================
add_library(peripherals STATIC
    peripherals/lsm6dsr.cpp
    peripherals/bm85163.cpp
    peripherals/qmc5883p.cpp
)
target_include_directories(peripherals PUBLIC
    ${PROJECT_INCLUDE_DIR}
)

# ==================== Knee 数据生产者库 ====================
add_library(knee_data_producer STATIC
    data_process/knee_data_producer.cpp
)
target_include_directories(knee_data_producer PUBLIC
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(knee_data_producer PUBLIC
    dsp_filters
    peripherals
    Threads::Threads
)

# ==================== Hip 数据生产者库 ====================
add_library(hip_data_producer STATIC
    data_process/hip_data_producer.cpp
)
target_include_directories(hip_data_producer PUBLIC
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(hip_data_producer PUBLIC
    dsp_filters
    peripherals
    Threads::Threads
)

# ==================== DSP 库函数 ====================
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/DspFilters SOURCE_LIB)
add_library(dsp_filters STATIC ${SOURCE_LIB})
target_include_directories(dsp_filters PUBLIC
    ${PROJECT_INCLUDE_DIR}
)
set_target_properties(dsp_filters PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ==================== Timing 工具库 ====================
add_library(timing_utils STATIC
    timing/periodic_timer.cpp
)
target_include_directories(timing_utils PUBLIC
    ${PROJECT_INCLUDE_DIR}
)