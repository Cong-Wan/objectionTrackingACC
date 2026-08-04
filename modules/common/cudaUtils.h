/*
Author: wilbur
Version: 1.0
Date: 2026-06-27
Description: CUDA 工具：错误检查、device buffer、pinned host buffer 与 event 计时器的 RAII 封装。
*/

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

void checkCuda(cudaError_t code, const std::string& message);

class DeviceBuffer {
public:
    DeviceBuffer();
    explicit DeviceBuffer(size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void allocate(size_t bytes);
    void* data();
    const void* data() const;
    size_t sizeBytes() const;

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
    void release();
};

class PinnedHostBuffer {
public:
    PinnedHostBuffer();
    explicit PinnedHostBuffer(size_t bytes);
    ~PinnedHostBuffer();
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void allocate(size_t bytes); // cudaMallocHost
    void* data();
    const void* data() const;
    size_t sizeBytes() const;

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
    void release();
};

class CudaEventTimer {
public:
    CudaEventTimer();
    ~CudaEventTimer();
    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void recordStart(cudaStream_t stream);
    void recordStop(cudaStream_t stream);
    float elapsedMs(); // cudaEventSynchronize(stop) 后返回 start->stop 毫秒

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
