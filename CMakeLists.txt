cmake_minimum_required(VERSION 3.10)

# Reset any toolchain
set(CMAKE_TOOLCHAIN_FILE "")

# Force compiler before project
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_GENERATOR "Unix Makefiles")

project(NeuralNetworkFromScratch C CXX)

set(CMAKE_CXX_STANDARD 17)

# Find and enable OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(neural_network
    main.cpp
    utils.cpp
    neural_network.cpp
    csv.cpp
)

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(neural_network OpenMP::OpenMP_CXX)
endif()