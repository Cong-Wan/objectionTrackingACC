#[[
Author: wilbur
Version: 1.1
Date: 2026-08-04
Description: CMake 构建脚本，定义 C++17/CUDA 工程、依赖路径与 trackerCpp/otaCore/trackCrop 目标。
    1.1: 新增 modules/encoder/nvencEncoder（NVENC SDK 直调硬编码 + libavformat mux），加入 otaCore 静态库，
    链接 libnvidia-encode，包含 /usr/local/include/ffnvcodec。
]]

cmake_minimum_required(VERSION 3.24)

# 强制使用 CUDA 12.6 的 nvcc（避免 PATH 中默认 nvcc 12.1 被误用），必须在 project() 之前设置。
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
endif()

project(objectionTrackingACC LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3090 = compute capability 8.6
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- 依赖路径默认值（可被 -D 覆盖）----
if(NOT TensorRT_ROOT)
    set(TensorRT_ROOT /home/Disk-6T/tensorrt)
endif()
if(NOT OpenCV_DIR)
    set(OpenCV_DIR /home/Disk-6T/opencv/build)
endif()
if(NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.6)
endif()

set(TENSORRT_INCLUDE_DIR ${TensorRT_ROOT}/include)
set(TENSORRT_LIB_DIR ${TensorRT_ROOT}/targets/x86_64-linux-gnu/lib)

# FFmpeg 头文件要求 __STDC_CONSTANT_MACROS/__STDC_FORMAT_MACROS
add_compile_definitions(__STDC_CONSTANT_MACROS __STDC_FORMAT_MACROS)

# ---- FFmpeg via pkg-config ----
find_package(PkgConfig REQUIRED)
if(DEFINED ENV{PKG_CONFIG_PATH})
    set(ENV{PKG_CONFIG_PATH} "/usr/local/ffmpeg7/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
else()
    set(ENV{PKG_CONFIG_PATH} "/usr/local/ffmpeg7/lib/pkgconfig")
endif()
pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil)
message(STATUS "Found FFmpeg: ${AVFORMAT_VERSION} (libavformat ${AVFORMAT_VERSION})")

# ---- OpenCV ----
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# ---- CUDA runtime ----
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA toolkit: ${CUDAToolkit_VERSION}")

# ---- Eigen3 / yaml-cpp（tracker 模块依赖）----
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
message(STATUS "Found Eigen3: ${Eigen3_VERSION}")
message(STATUS "Found yaml-cpp: ${yaml-cpp_VERSION}")

# ---- TensorRT（直接用已知路径）----
if(NOT EXISTS ${TENSORRT_INCLUDE_DIR})
    message(FATAL_ERROR "TensorRT include not found: ${TENSORRT_INCLUDE_DIR}")
endif()
if(NOT EXISTS ${TENSORRT_LIB_DIR})
    message(FATAL_ERROR "TensorRT lib dir not found: ${TENSORRT_LIB_DIR}")
endif()
message(STATUS "Found TensorRT: ${TensorRT_ROOT}")
include_directories(${TENSORRT_INCLUDE_DIR})

# ---- rpath：保证 build 树可执行文件运行时找到 CUDA 12.6 cudart、FFmpeg 与 TensorRT ----
set(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
set(CMAKE_BUILD_RPATH "${CUDA_LIB_DIR};/usr/local/ffmpeg7/lib;${TENSORRT_LIB_DIR}")
set(CMAKE_INSTALL_RPATH "${CUDA_LIB_DIR};/usr/local/ffmpeg7/lib;${TENSORRT_LIB_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# 可执行文件统一输出到 build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# otaCore 与 trackCrop 共享的链接库
set(OTA_LINK_LIBS
    PkgConfig::AVFORMAT
    PkgConfig::AVCODEC
    PkgConfig::AVUTIL
    ${OpenCV_LIBS}
    CUDA::cudart_static
    CUDA::cuda_driver
    ${TENSORRT_LIB_DIR}/libnvinfer.so)

# ---- trackerCpp 静态库（trackers_cpp 原样搬入，一行不许改）----
add_library(trackerCpp STATIC
    ${CMAKE_SOURCE_DIR}/modules/tracker/src/byteTracker.cpp
    ${CMAKE_SOURCE_DIR}/modules/tracker/src/baseTrack.cpp
    ${CMAKE_SOURCE_DIR}/modules/tracker/src/trackerConfig.cpp
    ${CMAKE_SOURCE_DIR}/modules/tracker/src/utils/kalmanFilter.cpp
    ${CMAKE_SOURCE_DIR}/modules/tracker/src/utils/matching.cpp)
target_include_directories(trackerCpp PUBLIC
    ${CMAKE_SOURCE_DIR}/modules/tracker/include
    ${CMAKE_SOURCE_DIR}/modules/tracker/include/utils)
target_link_libraries(trackerCpp PUBLIC Eigen3::Eigen yaml-cpp::yaml-cpp)

# ---- otaCore 静态库（modules/common + modules/decoder + modules/infer）----
add_library(otaCore STATIC
    ${CMAKE_SOURCE_DIR}/modules/common/frameTypes.cpp
    ${CMAKE_SOURCE_DIR}/modules/common/cudaUtils.cpp
    ${CMAKE_SOURCE_DIR}/modules/common/logger.cpp
    ${CMAKE_SOURCE_DIR}/modules/common/nv12ToBgr.cu
    ${CMAKE_SOURCE_DIR}/modules/decoder/nvdecDecoder.cpp
    ${CMAKE_SOURCE_DIR}/modules/encoder/nvencEncoder.cpp
    ${CMAKE_SOURCE_DIR}/modules/infer/gpuPreprocess.cu
    ${CMAKE_SOURCE_DIR}/modules/infer/trtInfer.cpp)
target_include_directories(otaCore PUBLIC
    ${CMAKE_SOURCE_DIR}/modules/common
    ${CMAKE_SOURCE_DIR}/modules/decoder
    ${CMAKE_SOURCE_DIR}/modules/encoder
    ${CMAKE_SOURCE_DIR}/modules/infer)
target_link_libraries(otaCore PUBLIC ${OTA_LINK_LIBS} nvidia-encode)

# ---- trackCrop ----
add_executable(trackCrop ${CMAKE_SOURCE_DIR}/apps/trackCrop/trackCropMain.cpp)
target_link_libraries(trackCrop PRIVATE otaCore trackerCpp)
