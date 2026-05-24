cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CUDACXX "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 90)
project(tiny-infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-G -g -DDEBUG")

find_package(CUDAToolkit REQUIRED)

add_executable(tiny-infer
    src/main.cu
    src/loader.cu
    src/kernels.cu
)

target_include_directories(tiny-infer PRIVATE src)
target_include_directories(tiny-infer PRIVATE include)

target_link_libraries(tiny-infer PRIVATE
    CUDA::cublas
    CUDA::cudart
)