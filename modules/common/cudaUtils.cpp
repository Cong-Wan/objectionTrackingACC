/*
Author: wilbur
Version: 1.0
Date: 2026-06-27
Description: CUDA 工具实现。
*/

#include "cudaUtils.h"
#include "logger.h"

#include <cstdint>
#include <stdexcept>
#include <string>

void checkCuda(cudaError_t code, const std::string& message) {
    if (code != cudaSuccess) {
        throw std::runtime_error(message + ": " + cudaGetErrorString(code) +
                                 " (code " + std::to_string(static_cast<int>(code)) + ")");
    }
}

// ---- DeviceBuffer ----
DeviceBuffer::DeviceBuffer() = default;

DeviceBuffer::DeviceBuffer(size_t bytes) {
    allocate(bytes);
}

DeviceBuffer::~DeviceBuffer() {
    release();
}

void DeviceBuffer::release() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
    bytes_ = 0;
}

void DeviceBuffer::allocate(size_t bytes) {
    release();
    if (bytes == 0) {
        return;
    }
    checkCuda(cudaMalloc(&ptr_, bytes), "cudaMalloc (DeviceBuffer)");
    bytes_ = bytes;
    logDebug("DeviceBuffer allocated " + std::to_string(bytes) + " bytes at " +
             std::to_string(reinterpret_cast<uintptr_t>(ptr_)));
}

void* DeviceBuffer::data() { return ptr_; }
const void* DeviceBuffer::data() const { return ptr_; }
size_t DeviceBuffer::sizeBytes() const { return bytes_; }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        release();
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

// ---- PinnedHostBuffer ----
PinnedHostBuffer::PinnedHostBuffer() = default;

PinnedHostBuffer::PinnedHostBuffer(size_t bytes) {
    allocate(bytes);
}

PinnedHostBuffer::~PinnedHostBuffer() {
    release();
}

void PinnedHostBuffer::release() {
    if (ptr_) {
        cudaFreeHost(ptr_);
        ptr_ = nullptr;
    }
    bytes_ = 0;
}

void PinnedHostBuffer::allocate(size_t bytes) {
    release();
    if (bytes == 0) {
        return;
    }
    checkCuda(cudaMallocHost(&ptr_, bytes), "cudaMallocHost (PinnedHostBuffer)");
    bytes_ = bytes;
    logDebug("PinnedHostBuffer allocated " + std::to_string(bytes) + " bytes at " +
             std::to_string(reinterpret_cast<uintptr_t>(ptr_)));
}

void* PinnedHostBuffer::data() { return ptr_; }
const void* PinnedHostBuffer::data() const { return ptr_; }
size_t PinnedHostBuffer::sizeBytes() const { return bytes_; }

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
        release();
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

// ---- CudaEventTimer ----
CudaEventTimer::CudaEventTimer() {
    checkCuda(cudaEventCreate(&start_), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
}

CudaEventTimer::~CudaEventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void CudaEventTimer::recordStart(cudaStream_t stream) {
    checkCuda(cudaEventRecord(start_, stream), "cudaEventRecord(start)");
}

void CudaEventTimer::recordStop(cudaStream_t stream) {
    checkCuda(cudaEventRecord(stop_, stream), "cudaEventRecord(stop)");
}

float CudaEventTimer::elapsedMs() {
    checkCuda(cudaEventSynchronize(stop_), "cudaEventSynchronize(stop)");
    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
    return ms;
}
