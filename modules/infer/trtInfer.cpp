/*
Author: wilbur
Version: 1.1
Date: 2026-08-04
Description: TensorRT 推理实现（batch=1）。load：engine 读取/反序列化/shape 校验/buffer 分配/stream 创建；
    infer：GpuFrame NV12 预处理直写 TRT 输入 → enqueueV3 → D2H pinned → 后处理
    （conf 过滤、letterbox 逆映射、clamp 到显示宽高、class-aware NMS、conf 降序限 maxDetections）。
    每次 infer 末尾 cudaStreamSynchronize，返回时结果已可用。
    1.1: 预处理调用传入 GpuFrame.fullRange，支持 limited-range 视频。
*/

#include "trtInfer.h"

#include "gpuPreprocess.h"
#include "logger.h"

#include <NvInfer.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        std::string text = std::string("[TensorRT] ") + (msg ? msg : "");
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                logError(text);
                break;
            case Severity::kWARNING:
                logWarn(text);
                break;
            case Severity::kINFO:
                if (isDebugEnabled()) {
                    logInfo(text);
                }
                break;
            default:
                if (isDebugEnabled()) {
                    logDebug(text);
                }
                break;
        }
    }
};

std::string dimsToString(const nvinfer1::Dims& dims) {
    std::string s = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        s += std::to_string(dims.d[i]);
        if (i + 1 < dims.nbDims) {
            s += ",";
        }
    }
    s += "]";
    return s;
}

struct Candidate {
    int classId;
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
};

float iou(const Candidate& a, const Candidate& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float uni = areaA + areaB - inter;
    if (uni <= 0.0f) {
        return 0.0f;
    }
    return inter / uni;
}

} // namespace

TrtInfer::TrtInfer(std::string enginePath, float confThreshold, float nmsIouThreshold, int maxDetections)
    : enginePath_(std::move(enginePath)),
      confThreshold_(confThreshold),
      nmsIouThreshold_(nmsIouThreshold),
      maxDetections_(maxDetections) {
    if (confThreshold_ <= 0.0f || confThreshold_ >= 1.0f) {
        throw std::runtime_error("TrtInfer confThreshold out of (0,1)");
    }
    if (nmsIouThreshold_ <= 0.0f || nmsIouThreshold_ >= 1.0f) {
        throw std::runtime_error("TrtInfer nmsIouThreshold out of (0,1)");
    }
    if (maxDetections_ < 1) {
        throw std::runtime_error("TrtInfer maxDetections must be >= 1");
    }
}

TrtInfer::~TrtInfer() {
    // 析构顺序：context → engine → runtime（runtime 必须最后释放）
    delete context_;
    delete engine_;
    delete runtime_;
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void TrtInfer::load() {
    // CUDA device/context 由解码器 primary_ctx=1 统一建立，这里不调用 cudaSetDevice。

    // 读取 engine 文件
    std::ifstream in(enginePath_, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open engine: " + enginePath_);
    }
    std::streamsize size = in.tellg();
    if (size <= 0) {
        throw std::runtime_error("Empty engine file: " + enginePath_);
    }
    in.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!in.read(data.data(), size)) {
        throw std::runtime_error("Failed to read engine: " + enginePath_);
    }
    logDebug("engine path=" + enginePath_ + " sizeBytes=" + std::to_string(data.size()));

    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        throw std::runtime_error("createInferRuntime failed");
    }
    logDebug("TensorRT runtime version: " + std::to_string(NV_TENSORRT_MAJOR) + "." +
             std::to_string(NV_TENSORRT_MINOR) + "." + std::to_string(NV_TENSORRT_PATCH));

    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!engine) {
        throw std::runtime_error("deserializeCudaEngine failed: " + enginePath_);
    }

    // 查找 input/output tensor 并校验
    bool foundInput = false;
    bool foundOutput = false;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (std::string(name) == kInputName) {
            foundInput = true;
            if (engine->getTensorDataType(kInputName) != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("engine input dtype is not FLOAT");
            }
            logDebug("input tensor=" + std::string(name) + " dtype=float shape=" +
                     dimsToString(engine->getTensorShape(kInputName)));
        } else if (std::string(name) == kOutputName) {
            foundOutput = true;
            logDebug("output tensor=" + std::string(name) + " dtype=" +
                     std::string(engine->getTensorDataType(kOutputName) == nvinfer1::DataType::kFLOAT
                                     ? "float"
                                     : "other") +
                     " shape=" + dimsToString(engine->getTensorShape(kOutputName)));
        }
    }
    if (!foundInput) {
        throw std::runtime_error("engine has no input tensor '" + std::string(kInputName) + "'");
    }
    if (!foundOutput) {
        throw std::runtime_error("engine has no output tensor '" + std::string(kOutputName) + "'");
    }

    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    nvinfer1::Dims4 inputDims;
    inputDims.nbDims = 4;
    inputDims.d[0] = 1;
    inputDims.d[1] = kChannelCount;
    inputDims.d[2] = kInputHeight;
    inputDims.d[3] = kInputWidth;
    if (!context->setInputShape(kInputName, inputDims)) {
        throw std::runtime_error("setInputShape failed for [1,3,640,640]");
    }
    nvinfer1::Dims outDims = context->getTensorShape(kOutputName);
    std::vector<int64_t> outShape;
    for (int i = 0; i < outDims.nbDims; ++i) {
        outShape.push_back(outDims.d[i]);
    }
    std::vector<int64_t> expected = {1, kOutputChannels, kAnchorCount};
    if (outShape != expected) {
        throw std::runtime_error("output shape mismatch: got " + dimsToString(outDims) +
                                 " expected [1,7,8400]");
    }
    logDebug("input shape=" + dimsToString(inputDims) + " output shape=" + dimsToString(outDims));

    // 分配 buffer
    size_t inputBytes = static_cast<size_t>(kChannelCount) * kInputWidth * kInputHeight * sizeof(float);
    size_t outputBytes = static_cast<size_t>(kOutputChannels) * kAnchorCount * sizeof(float);
    inputBuffer_.allocate(inputBytes);
    outputBuffer_.allocate(outputBytes);
    hostOutputBuffer_.allocate(outputBytes);
    logDebug("buffer bytes: input=" + std::to_string(inputBytes) + " output=" + std::to_string(outputBytes));

    if (!context->setTensorAddress(kInputName, inputBuffer_.data())) {
        throw std::runtime_error("setTensorAddress(input) failed");
    }
    if (!context->setTensorAddress(kOutputName, outputBuffer_.data())) {
        throw std::runtime_error("setTensorAddress(output) failed");
    }

    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    runtime_ = runtime.release();
    engine_ = engine.release();
    context_ = context.release();
    logInfo("TrtInfer loaded: " + enginePath_);
}

std::vector<Detection> TrtInfer::infer(const Frame& frame) {
    if (frame.device != Frame::Device::kCuda) {
        logWarn("TrtInfer: unsupported device, skip");
        return {};
    }
    if (!context_) {
        throw std::runtime_error("TrtInfer::infer called before load");
    }
    const GpuFrame& gpu = static_cast<const GpuFrame&>(frame);

    // GPU 预处理：NV12 -> NCHW RGB float32，零拷贝直写 TRT 输入
    LetterboxParams lb = computeLetterboxParams(gpu.width, gpu.height, kInputWidth, kInputHeight);
    preprocessNv12ToNchw(gpu.data0, gpu.data1, gpu.width, gpu.height,
                         gpu.linesize0, gpu.linesize1,
                         static_cast<float*>(inputBuffer_.data()),
                         0, 1, kInputWidth, kInputHeight, lb, gpu.fullRange, stream_);

    // 推理 + D2H
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("enqueueV3 failed");
    }
    checkCuda(cudaMemcpyAsync(hostOutputBuffer_.data(), outputBuffer_.data(),
                              outputBuffer_.sizeBytes(), cudaMemcpyDeviceToHost, stream_),
              "infer D2H copy");
    checkCuda(cudaStreamSynchronize(stream_), "infer cudaStreamSynchronize");

    return postProcess(lb, frame.frameId);
}

std::vector<Detection> TrtInfer::postProcess(const LetterboxParams& lb, int64_t frameId) {
    const int anchors = kAnchorCount;
    const float* frameOut = static_cast<const float*>(hostOutputBuffer_.data());

    std::vector<Candidate> candidates;
    for (int a = 0; a < anchors; ++a) {
        float cx = frameOut[0 * anchors + a];
        float cy = frameOut[1 * anchors + a];
        float w = frameOut[2 * anchors + a];
        float h = frameOut[3 * anchors + a];

        float bestScore = -1.0f;
        int bestClass = -1;
        for (int c = 0; c < kClassCount; ++c) {
            float score = frameOut[(4 + c) * anchors + a];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        if (bestClass < 0 || bestScore < confThreshold_) {
            continue;
        }

        // cxcywh -> xyxy，letterbox 逆映射，clamp 到显示宽高
        float x1L = cx - w / 2.0f;
        float y1L = cy - h / 2.0f;
        float x2L = cx + w / 2.0f;
        float y2L = cy + h / 2.0f;
        float x1 = (x1L - lb.padX) / lb.scale;
        float y1 = (y1L - lb.padY) / lb.scale;
        float x2 = (x2L - lb.padX) / lb.scale;
        float y2 = (y2L - lb.padY) / lb.scale;
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(lb.originalWidth)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(lb.originalHeight)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(lb.originalWidth)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(lb.originalHeight)));

        candidates.push_back({bestClass, bestScore, x1, y1, x2, y2});
    }

    // class-aware NMS
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& l, const Candidate& r) { return l.confidence > r.confidence; });
    std::vector<char> suppressed(candidates.size(), 0);
    std::vector<Detection> detections;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        Detection d;
        d.frameId = frameId;
        d.classId = candidates[i].classId;
        d.confidence = candidates[i].confidence;
        d.x1 = candidates[i].x1;
        d.y1 = candidates[i].y1;
        d.x2 = candidates[i].x2;
        d.y2 = candidates[i].y2;
        detections.push_back(d);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j] || candidates[j].classId != candidates[i].classId) {
                continue;
            }
            if (iou(candidates[i], candidates[j]) > nmsIouThreshold_) {
                suppressed[j] = 1;
            }
        }
    }

    // 按 conf 降序限 maxDetections（NMS 输入已降序，detections 保持降序）
    if (static_cast<int>(detections.size()) > maxDetections_) {
        detections.resize(maxDetections_);
    }
    return detections;
}
