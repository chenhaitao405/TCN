cmake_minimum_required(VERSION 3.0)

project(DSPFilters CXX)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/source SOURCE_LIB)

add_library(dsp_filters STATIC ${SOURCE_LIB})

target_include_directories(dsp_filters PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(dsp_filters PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

