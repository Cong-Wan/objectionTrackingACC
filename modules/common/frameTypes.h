/*
Author: wilbur
Version: 1.1
Date: 2026-08-04
Description: 帧与数据传递类型定义。
    - Frame：帧抽象基类（device/pixFmt 标识，解码/推理接口只依赖它）。
    - GpuFrame：NVDEC 输出的 NV12 GPU 帧（AVFrame 引用 + plane 指针/linesize）。
    - Detection：单个检测结果（原图坐标系 xyxy）。
    - LetterboxParams：letterbox 缩放参数，预处理填充，后处理用于坐标还原。
    （本头文件禁止 include 任何 FFmpeg 头，AVFrame 仅前向声明）
    1.1: Frame 增加 fullRange 标识（NV12 色彩量程），支持 limited-range 视频。
*/

#pragma once

#include <cstdint>
#include <memory>

struct Frame {
    enum class Device { kCuda, kDrm };          // kDrm 预留给 RK
    enum class PixFmt { kNv12, kNv21, kRgb };
    int64_t frameId = 0;
    int width = 0;                              // 显示宽（非 stride 对齐）
    int height = 0;
    Device device = Device::kCuda;
    PixFmt pixFmt = PixFmt::kNv12;
    bool fullRange = true;                      // NV12 量程：true=full(pc)，false=limited(tv)
    virtual ~Frame() = default;
};

struct AVFrame;                                 // 前向声明！本头文件禁止 include 任何 FFmpeg 头

struct GpuFrame : Frame {
    uint8_t* data0 = nullptr;                   // Y plane
    uint8_t* data1 = nullptr;                   // UV plane
    int linesize0 = 0;
    int linesize1 = 0;
    // 注意：shared_ptr 默认 deleter 是 delete，AVFrame 必须 av_frame_free！
    // 构造一律照搬 videoDecoder 原写法：shared_ptr<AVFrame>(p, [](AVFrame* f){ if(f) av_frame_free(&f); })
    std::shared_ptr<AVFrame> frameRef;
    ~GpuFrame() override;                       // 定义在 frameTypes.cpp（AVFrame 完整类型可见处）
};

struct Detection {                              // 原图坐标系 xyxy
    int64_t frameId = 0;
    int classId = 0;
    float confidence = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct LetterboxParams {
    int originalWidth = 0;
    int originalHeight = 0;
    int inputWidth = 0;
    int inputHeight = 0;
    float scale = 0.0f;
    int resizedWidth = 0;
    int resizedHeight = 0;
    int padX = 0;
    int padY = 0;
};
