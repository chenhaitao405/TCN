# ==================== NPY 推理工具 ====================
add_executable(npy_infer npy_infer.cpp)
target_include_directories(npy_infer PRIVATE
    ${RKNN_API_PATH}/include
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(npy_infer
    ${RKNN_RT_LIB}
)

add_executable(pattern_npy_infer pattern_npy_infer.cpp)
target_include_directories(pattern_npy_infer PRIVATE
    ${RKNN_API_PATH}/include
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(pattern_npy_infer
    ${RKNN_RT_LIB}
)

add_executable(pattern_npy_infer_cb pattern_npy_infer_cb.cpp)
target_include_directories(pattern_npy_infer_cb PRIVATE
    ${RKNN_API_PATH}/include
    ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(pattern_npy_infer_cb
    ${RKNN_RT_LIB}
)

install(TARGETS npy_infer pattern_npy_infer pattern_npy_infer_cb DESTINATION npy_infer)
