cmake_minimum_required(VERSION 3.10)
project(state_detection)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_definitions(-D_USE_MATH_DEFINES=True)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
else()
    set(CMAKE_CXX_FLAGS "-Wno-unused-result")
endif()

# 自动包含 src 目录及其子目录中的所有 .cpp 文件
file(GLOB_RECURSE SOURCE
    src/*.cpp
)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

include_directories(ThirdParty/eigen-3.3.9)
include_directories(ThirdParty/sophus)
include_directories(ThirdParty/dj_fft)

# Add Sophus library
# add_subdirectory(ThirdParty/sophus)
# target_link_libraries(${PROJECT_NAME} sophus)

# 将所有源文件添加到可执行文件中
add_executable(${PROJECT_NAME} ${SOURCE})
