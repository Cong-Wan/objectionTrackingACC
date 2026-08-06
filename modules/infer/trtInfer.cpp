/*
Author: wilbur
Version: 1.2
Date: 2026-08-06
Description: TensorRT 推理实现（batch=1）。load：engine 读取/反序列化/shape 校验/buffer 分配/stream 创建；
    infer：GpuFrame NV12 预处理直写 TRT 输入 → enqueueV3 → D2H pinned → 后处理
    （conf 过滤、letterbox 逆映射、clamp 到显示宽高、class-aware NMS、conf 降序限 maxDetections）。
    每次 infer 末尾 cudaStreamSynchronize，返回时结果已可用。
    1.1: 预处理调用传入 GpuFrame.fullRange，支持 limited-range 视频。
    1.2: 兼容 [u32 metaLen][json][ftrt plan] 封装；输入 H/W、输出 anchors/layout/numClasses 从 engine 读取，
    废除 640/8400 硬编码；numClasses 不强制为 3。
*/

#include "trtInfer.h"

#include "gpuPreprocess.h"
#include "logger.h"

#include <NvInfer.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
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

bool startsWithFtrt(const char* p, size_t n) {
    return n >= 4 && std::memcmp(p, "ftrt", 4) == 0;
}

bool isLikelyJsonObject(const char* p, size_t n) {
    if (n < 2) return false;
    size_t b = 0;
    while (b < n && std::isspace(static_cast<unsigned char>(p[b]))) ++b;
    size_t e = n;
    while (e > b && std::isspace(static_cast<unsigned char>(p[e - 1]))) --e;
    if (e <= b) return false;
    return p[b] == '{' && p[e - 1] == '}';
}

// 从 engine 文件中剥离可选元数据头，返回裸 plan 偏移与长度
struct EnginePlanSlice {
    size_t planOffset = 0;
    size_t planSize = 0;
    std::string metaJson;  // 可空
};

EnginePlanSlice extractEnginePlan(const std::vector<char>& data) {
    if (data.empty()) {
        throw std::runtime_error("empty engine blob");
    }
    EnginePlanSlice out;

    // 1) 裸 plan
    if (startsWithFtrt(data.data(), data.size())) {
        out.planOffset = 0;
        out.planSize = data.size();
        return out;
    }

    // 2) [u32le metaLen][json][plan]
    if (data.size() >= 8) {
        uint32_t metaLen = 0;
        std::memcpy(&metaLen, data.data(), sizeof(metaLen));
        const size_t maxMeta = std::min<size_t>(1u << 20, data.size() > 8 ? data.size() - 8 : 0);
        if (metaLen >= 2 && static_cast<size_t>(metaLen) <= maxMeta) {
            const size_t jsonOff = 4;
            const size_t planOff = 4 + static_cast<size_t>(metaLen);
            if (planOff + 4 <= data.size() &&
                isLikelyJsonObject(data.data() + jsonOff, metaLen) &&
                startsWithFtrt(data.data() + planOff, data.size() - planOff)) {
                out.planOffset = planOff;
                out.planSize = data.size() - planOff;
                out.metaJson.assign(data.data() + jsonOff, data.data() + jsonOff + metaLen);
                return out;
            }
        }
    }

    // 3) 兜底：搜索 ftrt
    for (size_t i = 0; i + 4 <= data.size(); ++i) {
        if (startsWithFtrt(data.data() + i, data.size() - i)) {
            out.planOffset = i;
            out.planSize = data.size() - i;
            if (i >= 2 && isLikelyJsonObject(data.data(), i)) {
                out.metaJson.assign(data.data(), data.data() + i);
            } else if (i > 4) {
                // 可能是 u32 + json
                uint32_t metaLen = 0;
                std::memcpy(&metaLen, data.data(), sizeof(metaLen));
                if (static_cast<size_t>(metaLen) + 4 == i &&
                    isLikelyJsonObject(data.data() + 4, metaLen)) {
                    out.metaJson.assign(data.data() + 4, data.data() + 4 + metaLen);
                }
            }
            return out;
        }
    }

    throw std::runtime_error("unrecognized engine format: no ftrt plan magic found");
}

// 从 meta JSON 中尽力抽取 imgsz（仅日志/动态维兜底，不覆盖 engine shape）
bool tryParseImgsz(const std::string& json, int& h, int& w) {
    h = w = 0;
    auto pos = json.find("\"imgsz\"");
    if (pos == std::string::npos) return false;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return false;
    size_t end = json.find(']', pos);
    if (end == std::string::npos) return false;
    std::string arr = json.substr(pos + 1, end - pos - 1);
    // 允许 "1280, 1280" 或换行
    int vals[2] = {0, 0};
    int n = 0;
    size_t i = 0;
    while (i < arr.size() && n < 2) {
        while (i < arr.size() && (std::isspace(static_cast<unsigned char>(arr[i])) || arr[i] == ',')) ++i;
        if (i >= arr.size()) break;
        size_t j = i;
        while (j < arr.size() && (std::isdigit(static_cast<unsigned char>(arr[j])))) ++j;
        if (j == i) break;
        vals[n++] = std::stoi(arr.substr(i, j - i));
        i = j;
    }
    if (n == 1) {
        h = w = vals[0];
        return h > 0;
    }
    if (n >= 2) {
        // 常见 [w,h] 或 [h,w]；Ultralytics 多为 [h,w] 同值；若两值不同按 [h,w]=imgsz 约定
        h = vals[0];
        w = vals[1];
        return h > 0 && w > 0;
    }
    return false;
}

std::string metaSummary(const std::string& json) {
    if (json.empty()) return "(none)";
    // 压缩空白，截断
    std::string s;
    s.reserve(std::min<size_t>(json.size(), 256));
    bool sp = false;
    for (char c : json) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!sp && !s.empty()) {
                s.push_back(' ');
                sp = true;
            }
            continue;
        }
        sp = false;
        s.push_back(c);
        if (s.size() >= 240) {
            s += "...";
            break;
        }
    }
    return s;
}

size_t dtypeBytes(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        default: return 4;
    }
}

// layoutA: [C,A] ch-major; layoutB: [A,C] anchor-major
inline float atOut(const float* p, int ch, int a, int anchors, int outCh, bool layoutA) {
    return layoutA ? p[ch * anchors + a] : p[a * outCh + ch];
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
    delete context_;
    delete engine_;
    delete runtime_;
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void TrtInfer::load() {
    // CUDA device/context 由解码器 primary_ctx=1 统一建立，这里不调用 cudaSetDevice。

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

    EnginePlanSlice slice = extractEnginePlan(data);
    if (slice.metaJson.empty()) {
        logDebug("engine file format=raw-plan planOffset=0 planBytes=" +
                 std::to_string(slice.planSize));
    } else {
        logDebug("engine file format=meta+plan metaLen=" +
                 std::to_string(slice.metaJson.size()) + " planOffset=" +
                 std::to_string(slice.planOffset) + " planBytes=" +
                 std::to_string(slice.planSize));
        logDebug("engine meta: " + metaSummary(slice.metaJson));
    }

    int metaH = 0, metaW = 0;
    const bool hasMetaImgsz = tryParseImgsz(slice.metaJson, metaH, metaW);
    if (hasMetaImgsz) {
        logDebug("engine meta imgsz=" + std::to_string(metaH) + "x" + std::to_string(metaW));
    }

    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        throw std::runtime_error("createInferRuntime failed");
    }
    logDebug("TensorRT runtime version: " + std::to_string(NV_TENSORRT_MAJOR) + "." +
             std::to_string(NV_TENSORRT_MINOR) + "." + std::to_string(NV_TENSORRT_PATCH));

    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(data.data() + slice.planOffset, slice.planSize));
    if (!engine) {
        throw std::runtime_error("deserializeCudaEngine failed: " + enginePath_ +
                                 " (planOffset=" + std::to_string(slice.planOffset) +
                                 ", planBytes=" + std::to_string(slice.planSize) + ")");
    }

    // 发现 input / output tensor
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (!name) continue;
        const auto mode = engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputNames.emplace_back(name);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            outputNames.emplace_back(name);
        }
        logDebug(std::string("tensor ") + name + " mode=" +
                 (mode == nvinfer1::TensorIOMode::kINPUT
                      ? "INPUT"
                      : (mode == nvinfer1::TensorIOMode::kOUTPUT ? "OUTPUT" : "OTHER")) +
                 " dtype=" +
                 (engine->getTensorDataType(name) == nvinfer1::DataType::kFLOAT ? "float" : "other") +
                 " shape=" + dimsToString(engine->getTensorShape(name)));
    }
    if (inputNames.empty()) {
        throw std::runtime_error("engine has no INPUT tensor");
    }
    if (outputNames.empty()) {
        throw std::runtime_error("engine has no OUTPUT tensor");
    }

    auto pickName = [](const std::vector<std::string>& names, const char* preferred) {
        for (const auto& n : names) {
            if (n == preferred) return n;
        }
        return names.front();
    };
    inputName_ = pickName(inputNames, "images");
    outputName_ = pickName(outputNames, "output0");
    if (inputNames.size() > 1) {
        logWarn("engine has multiple inputs, using '" + inputName_ + "'");
    }
    if (outputNames.size() > 1) {
        logWarn("engine has multiple outputs, using '" + outputName_ + "'");
    }

    if (engine->getTensorDataType(inputName_.c_str()) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("engine input dtype is not FLOAT: " + inputName_);
    }
    if (engine->getTensorDataType(outputName_.c_str()) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("engine output dtype is not FLOAT: " + outputName_);
    }

    nvinfer1::Dims inShape = engine->getTensorShape(inputName_.c_str());
    if (inShape.nbDims != 4) {
        throw std::runtime_error("engine input rank must be 4 NCHW, got " + dimsToString(inShape));
    }

    // 解析 / 设定 [N,C,H,W]
    int n = static_cast<int>(inShape.d[0]);
    int c = static_cast<int>(inShape.d[1]);
    int h = static_cast<int>(inShape.d[2]);
    int w = static_cast<int>(inShape.d[3]);
    if (n != 1 && n != -1) {
        throw std::runtime_error("only batch=1 supported, engine N=" + std::to_string(n));
    }
    if (c != 3 && c != -1) {
        throw std::runtime_error("input channels must be 3, got " + std::to_string(c));
    }
    if (h <= 0 || w <= 0) {
        if (hasMetaImgsz) {
            h = metaH;
            w = metaW;
            logWarn("input H/W dynamic in engine, fallback to meta imgsz=" + std::to_string(h) +
                    "x" + std::to_string(w));
        } else {
            throw std::runtime_error("input H/W dynamic and no meta imgsz; rebuild engine with fixed shape");
        }
    }
    inputH_ = h;
    inputW_ = w;

    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("createExecutionContext failed");
    }

    nvinfer1::Dims4 inputDims;
    inputDims.nbDims = 4;
    inputDims.d[0] = 1;
    inputDims.d[1] = 3;
    inputDims.d[2] = inputH_;
    inputDims.d[3] = inputW_;
    if (!context->setInputShape(inputName_.c_str(), inputDims)) {
        throw std::runtime_error("setInputShape failed for [1,3," + std::to_string(inputH_) + "," +
                                 std::to_string(inputW_) + "] tensor=" + inputName_);
    }

    nvinfer1::Dims outDims = context->getTensorShape(outputName_.c_str());
    if (outDims.nbDims != 3) {
        throw std::runtime_error("engine output rank must be 3, got " + dimsToString(outDims));
    }
    const int d0 = static_cast<int>(outDims.d[0]);
    const int d1 = static_cast<int>(outDims.d[1]);
    const int d2 = static_cast<int>(outDims.d[2]);
    if (d0 != 1 && d0 != -1) {
        throw std::runtime_error("output batch must be 1, got " + dimsToString(outDims));
    }
    if (d1 <= 0 || d2 <= 0) {
        throw std::runtime_error("output shape incomplete after setInputShape: " + dimsToString(outDims));
    }

    // 布局判定：通道维应明显小于锚点维
    if (d1 < d2 && d1 >= 5) {
        // [1, C, A]
        outputLayoutA_ = true;
        outChannels_ = d1;
        numAnchors_ = d2;
    } else if (d2 < d1 && d2 >= 5) {
        // [1, A, C]
        outputLayoutA_ = false;
        outChannels_ = d2;
        numAnchors_ = d1;
    } else {
        throw std::runtime_error("unrecognized YOLO output layout: " + dimsToString(outDims) +
                                 " (expect [1,4+nc,anchors] or [1,anchors,4+nc])");
    }
    numClasses_ = outChannels_ - 4;
    if (numClasses_ < 1) {
        throw std::runtime_error("numClasses=outChannels-4 must be >=1, outChannels=" +
                                 std::to_string(outChannels_));
    }

    logDebug("input tensor=" + inputName_ + " dtype=float shape=[1,3," + std::to_string(inputH_) +
             "," + std::to_string(inputW_) + "]");
    logDebug("output tensor=" + outputName_ + " dtype=float shape=" + dimsToString(outDims) +
             " layout=" + std::string(outputLayoutA_ ? "A[1,C,A]" : "B[1,A,C]") +
             " outChannels=" + std::to_string(outChannels_) +
             " anchors=" + std::to_string(numAnchors_) +
             " numClasses=" + std::to_string(numClasses_));

    // 分配 buffer
    const size_t inputBytes =
        static_cast<size_t>(3) * static_cast<size_t>(inputH_) * static_cast<size_t>(inputW_) * sizeof(float);
    size_t outElems = 1;
    for (int i = 0; i < outDims.nbDims; ++i) {
        if (outDims.d[i] > 0) outElems *= static_cast<size_t>(outDims.d[i]);
    }
    const size_t outputBytes = outElems * dtypeBytes(engine->getTensorDataType(outputName_.c_str()));
    inputBuffer_.allocate(inputBytes);
    outputBuffer_.allocate(outputBytes);
    hostOutputBuffer_.allocate(outputBytes);
    logDebug("buffer bytes: input=" + std::to_string(inputBytes) + " output=" + std::to_string(outputBytes));

    if (!context->setTensorAddress(inputName_.c_str(), inputBuffer_.data())) {
        throw std::runtime_error("setTensorAddress(input) failed");
    }
    if (!context->setTensorAddress(outputName_.c_str(), outputBuffer_.data())) {
        throw std::runtime_error("setTensorAddress(output) failed");
    }

    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    runtime_ = runtime.release();
    engine_ = engine.release();
    context_ = context.release();
    logInfo("TrtInfer loaded: " + enginePath_ + " input=" + std::to_string(inputW_) + "x" +
            std::to_string(inputH_) + " anchors=" + std::to_string(numAnchors_) +
            " classes=" + std::to_string(numClasses_));
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

    LetterboxParams lb = computeLetterboxParams(gpu.width, gpu.height, inputW_, inputH_);
    preprocessNv12ToNchw(gpu.data0, gpu.data1, gpu.width, gpu.height,
                         gpu.linesize0, gpu.linesize1,
                         static_cast<float*>(inputBuffer_.data()),
                         0, 1, inputW_, inputH_, lb, gpu.fullRange, stream_);

    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("enqueueV3 failed");
    }
    checkCuda(cudaMemcpyAsync(hostOutputBuffer_.data(), outputBuffer_.data(),
                              outputBuffer_.sizeBytes(), cudaMemcpyDeviceToHost, stream_),
              "infer D2H copy");
    checkCuda(cudaStreamSynchronize(stream_), "infer cudaStreamSynchronize");

    return postProcess(lb, frame.frameId);
}

std::vector<Detection> TrtInfer::postProcess(const LetterboxParams& lb, int64_t frameId) const {
    const int anchors = numAnchors_;
    const int outCh = outChannels_;
    const int ncls = numClasses_;
    const bool layoutA = outputLayoutA_;
    const float* frameOut = static_cast<const float*>(hostOutputBuffer_.data());

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(std::min(anchors, maxDetections_ * 8)));

    for (int a = 0; a < anchors; ++a) {
        const float cx = atOut(frameOut, 0, a, anchors, outCh, layoutA);
        const float cy = atOut(frameOut, 1, a, anchors, outCh, layoutA);
        const float bw = atOut(frameOut, 2, a, anchors, outCh, layoutA);
        const float bh = atOut(frameOut, 3, a, anchors, outCh, layoutA);

        float bestScore = -1.0f;
        int bestClass = -1;
        for (int c = 0; c < ncls; ++c) {
            const float score = atOut(frameOut, 4 + c, a, anchors, outCh, layoutA);
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }
        if (bestClass < 0 || bestScore < confThreshold_) {
            continue;
        }

        float x1L = cx - bw / 2.0f;
        float y1L = cy - bh / 2.0f;
        float x2L = cx + bw / 2.0f;
        float y2L = cy + bh / 2.0f;
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

    if (static_cast<int>(detections.size()) > maxDetections_) {
        detections.resize(maxDetections_);
    }
    return detections;
}
