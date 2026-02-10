cmake_minimum_required(VERSION 3.15)

# Project name and version
project(spm-hardware-sim LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Create the executable
add_executable(spm-simulator
    src/main.cpp
)

# Include directories
target_include_directories(spm-simulator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compiler-specific options
if(MSVC)
    # Microsoft Visual C++ options
    target_compile_options(spm-simulator PRIVATE
        /W4
    )
else()
    # GCC/Clang options
    target_compile_options(spm-simulator PRIVATE
        -Wall
        -Wextra
        -pedantic
    )
endif()

# Create a library for the SPM components
add_library(spm-hardware-sim-lib
    src/main.cpp
)

# Install targets
install(TARGETS spm-simulator
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Add custom targets for convenience
add_custom_target(run
    COMMAND $<TARGET_FILE:spm-simulator>
    COMMENT "Running SPM Simulator"
)