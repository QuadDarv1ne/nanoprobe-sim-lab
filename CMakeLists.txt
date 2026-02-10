cmake_minimum_required(VERSION 3.15)

# Project name and version
project(NanoprobeSimulationLab VERSION 1.0.0 LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add subdirectories for each component
add_subdirectory(cpp-spm-hardware-sim)

# Define main executable that integrates all components
add_executable(nanoprobe-main
    main.cpp  # This would be created separately
)

# Include directories
target_include_directories(nanoprobe-main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp-spm-hardware-sim/include
    ${CMAKE_CURRENT_SOURCE_DIR}/py-surface-image-analyzer/src
    ${CMAKE_CURRENT_SOURCE_DIR}/py-sstv-groundstation/src
)

# Link libraries from subprojects
target_link_libraries(nanoprobe-main
    spm-hardware-sim-lib  # From cpp-spm-hardware-sim
)

# Install targets
install(TARGETS nanoprobe-main
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Add custom targets for convenience
add_custom_target(run-all
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/nanoprobe-main
    COMMENT "Running Nanoprobe Simulation Lab"
)

add_custom_target(docs
    COMMAND doxygen Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating documentation"
)

# Enable testing
enable_testing()
add_subdirectory(tests)