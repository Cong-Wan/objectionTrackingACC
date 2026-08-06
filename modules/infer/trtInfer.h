/*
Author: wilbur
Version: 1.1
Date: 2026-08-06
Description: TensorRT 推理引擎（inferRunner + postProcessor 合并，batch 恒为 1）。
    加载 engine、NV12 零拷贝预处理直写输入、enqueueV3 推理、YOLO detect 输出解析
    （conf 过滤 + letterbox 逆映射 + clamp + class-aware NMS）。
    1.1: 兼容 meta+plan 封装 engine；输入/输出 shape、锚点数、类别数从 engine 读取，废除 640/8400 硬编码。
    应用层按 classId 规则取用（0=body 1=head 2=upperBody 等），不强制 numClasses==3。
*/

#pragma once

#include "cudaUtils.h"
#include "frameTypes.h"
#include "inferBase.h"

#include <cuda_runtime.h>

#include <cstddef>
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

    int inputWidth() const { return inputW_; }
    int inputHeight() const { return inputH_; }
    int numClasses() const { return numClasses_; }
    int numAnchors() const { return numAnchors_; }
    const std::string& inputName() const { return inputName_; }
    const std::string& outputName() const { return outputName_; }

private:
    std::vector<Detection> postProcess(const LetterboxParams& lb, int64_t frameId) const;

    std::string enginePath_;
    float confThreshold_;
    float nmsIouThreshold_;
    int maxDetections_;

    // load() 后由 engine 填充
    std::string inputName_;
    std::string outputName_;
    int inputW_ = 0;
    int inputH_ = 0;
    int outChannels_ = 0;   // 4 + numClasses
    int numAnchors_ = 0;
    int numClasses_ = 0;
    bool outputLayoutA_ = true;  // true: [1,C,A]  false: [1,A,C]

    DeviceBuffer inputBuffer_;
    DeviceBuffer outputBuffer_;
    PinnedHostBuffer hostOutputBuffer_;
    cudaStream_t stream_ = nullptr;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
};
