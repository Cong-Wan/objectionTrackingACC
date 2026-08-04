/*
Author: wilbur
Version: 1.0
Date: 2026-08-04
Description: FFmpeg CUDA/NVDEC 视频解码器。打开视频，输出 NV12 GPU 帧（GpuFrame），
    校验 sw_format / color range / color space，并维护 AVFrame 引用直到下游处理完成。
*/

#pragma once

#include "decoderBase.h"

#include <memory>
#include <string>

struct AVFormatContext;
struct AVCodecContext;
struct AVBufferRef;

class NvdecDecoder : public IVideoDecoder {
public:
    // sharedHwDeviceCtx 非空时复用该 CUDA hw device context（保证与 GPU 预处理/TRT 同一 context）；
    // 为空时自行创建（device 0 primary context）。
    explicit NvdecDecoder(std::string videoPath, AVBufferRef* sharedHwDeviceCtx = nullptr);
    ~NvdecDecoder() override;

    NvdecDecoder(const NvdecDecoder&) = delete;
    NvdecDecoder& operator=(const NvdecDecoder&) = delete;

    void open() override;
    bool readFrame(std::unique_ptr<Frame>& frame) override;

    int width() const override;
    int height() const override;
    double fps() const override;

private:
    std::string videoPath_;
    AVBufferRef* sharedHwDeviceCtx_ = nullptr;

    AVFormatContext* formatCtx_ = nullptr;
    AVCodecContext* codecCtx_ = nullptr;
    AVBufferRef* hwDeviceCtx_ = nullptr;
    int streamIndex_ = -1;

    int64_t nextFrameId_ = 0;
    bool flushing_ = false;
};
