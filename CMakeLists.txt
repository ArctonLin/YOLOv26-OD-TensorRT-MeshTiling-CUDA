cmake_minimum_required(VERSION 3.10)
project(YOLOTensorRT_MeshTiling)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-arch=native)

# Find OpenCV with CUDA support
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# TensorRT (usually in standard paths or needs specific search)
find_path(TRT_INCLUDE_DIR NvInfer.h)
find_library(TRT_LIBRARY nvinfer)
find_library(TRT_PLUGIN_LIBRARY nvinfer_plugin)

# JSON parser (nlohmann_json) - Assuming we will download or use it
find_package(nlohmann_json QUIET)
if (NOT nlohmann_json_FOUND)
    # Fallback if not installed globally
    include(FetchContent)
    FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
    FetchContent_MakeAvailable(json)
endif()

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

cuda_add_library(cuda_kernels src/cuda_kernels.cu)

add_executable(yolo_detector src/main.cpp src/yolo_detector.cpp)

target_link_libraries(yolo_detector 
    cuda_kernels
    ${OpenCV_LIBS}
    ${TRT_LIBRARY} 
    ${TRT_PLUGIN_LIBRARY}
    ${CUDA_LIBRARIES}
    nlohmann_json::nlohmann_json
    pthread
)
