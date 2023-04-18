cmake_minimum_required(VERSION 3.25)
project(cuda_examples CUDA)

set(CMAKE_CUDA_STANDARD 20)
find_package(CUDA REQUIRED)

#set(CMAKE_CXX_STANDARD 11)

# Set the CUDA architecture to use
set(CUDA_ARCH "-arch=sm_75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

# Add the CUBLAS library
set(CUFFT_LIBRARIES cufft)

#add_library (parallel_fft_lib parallel_fft.cu)
add_executable(fft main.cu)

# Link against CUBLAS and the CUDA libraries
target_link_libraries(fft ${CUFFT_LIBRARIES} ${CUDA_LIBRARIES})

# Set the CUDA flags for the executable
set_target_properties(
    fft PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

# Set the optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()