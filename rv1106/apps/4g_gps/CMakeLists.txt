# ==================== 4G GPS 程序 ====================
add_executable(serial2nt26 serial2nt26.cpp)
target_include_directories(serial2nt26 PRIVATE
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(serial2nt26
    peripherals
    Threads::Threads
)

install(TARGETS serial2nt26 DESTINATION 4g_gps)
