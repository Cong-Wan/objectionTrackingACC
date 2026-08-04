/*
Author: wilbur
Version: 1.0
Date: 2026-08-04
Description: TensorRT 推理引擎（inferRunner + postProcessor 合并，batch 恒为 1）。
    加载 engine、校验 input [1,3,640,640] / output [1,7,8400]、NV12 零拷贝预处理直写输入、
    enqueueV3 推理、output0 解析（conf 过滤 + letterbox 逆映射 + clamp + class-aware NMS）。
    模型常量：输入 640x640、输出 7x8400（cxcywh + 3 类分数）、类别 0=fullBody 1=head 2=upperBody。
*/

#pragma once

#include "cudaUtils.h"
#include "frameTypes.h"
#include "inferBase.h"

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

class TrtInfer : public IInferEngine {
public:
    TrtInfer(std::string enginePath, float confThreshold, float nmsIouThreshold = 0.45f,
             int maxDetections = 300);
    ~TrtInfer() override;

    TrtInfer(const TrtInfer&) = delete;
    TrtInfer& operator=(const TrtInfer&) = delete;

    void load() override;
    std::vector<Detection> infer(const Frame& frame) override;

private:
    std::vector<Detection> postProcess(const LetterboxParams& lb, int64_t frameId);

    // 模型常量（batch1.int8.engine 已确认事实）
    static constexpr int kInputWidth = 640;
    static constexpr int kInputHeight = 640;
    static constexpr int kChannelCount = 3;
    static constexpr int kOutputChannels = 7;
    static constexpr int kAnchorCount = 8400;
    static constexpr int kClassCount = 3;
    static constexpr const char* kInputName = "images";
    static constexpr const char* kOutputName = "output0";

    std::string enginePath_;
    float confThreshold_;
    float nmsIouThreshold_;
    int maxDetections_;

    DeviceBuffer inputBuffer_;
    DeviceBuffer outputBuffer_;
    PinnedHostBuffer hostOutputBuffer_;
    cudaStream_t stream_ = nullptr;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
};
