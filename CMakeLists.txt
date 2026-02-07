cmake_minimum_required(VERSION 3.21)
project(smll)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Option to build llama.cpp
option(SMLL_BUILD_LLAMA "Build llama.cpp and link against it" ON)

# Add llama.cpp as a subdirectory if enabled
if(SMLL_BUILD_LLAMA)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
    add_subdirectory(vendor/llama.cpp)
endif()

# Create the Python extension module
pybind11_add_module(_smll include/smll.cpp)

# Include directories
target_include_directories(_smll PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/ggml/include
)

# Link against llama.cpp if built
if(SMLL_BUILD_LLAMA)
    target_link_libraries(_smll PRIVATE llama)
endif()

# Set C++ standard
target_compile_features(_smll PRIVATE cxx_std_11)

# Install the extension module
install(TARGETS _smll LIBRARY DESTINATION smll)
