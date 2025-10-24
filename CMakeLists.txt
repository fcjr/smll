cmake_minimum_required(VERSION 3.21)
project(smol)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Option to build llama.cpp
option(SMOL_BUILD_LLAMA "Build llama.cpp and link against it" ON)

# Add llama.cpp as a subdirectory if enabled
if(SMOL_BUILD_LLAMA)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(vendor/llama.cpp)
endif()

# Create the Python extension module
pybind11_add_module(_smol include/smol.cpp)

# Include directories
target_include_directories(_smol PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/ggml/include
)

# Link against llama.cpp if built
if(SMOL_BUILD_LLAMA)
    target_link_libraries(_smol PRIVATE llama)
endif()

# Set C++ standard
target_compile_features(_smol PRIVATE cxx_std_11)

# Install the extension module
install(TARGETS _smol LIBRARY DESTINATION smol)
