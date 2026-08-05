/*
Author: wilbur
Version: 1.0
Date: 2026-08-05
Description: NVENC 硬编码视频编码器（H.264）。直接调用 NVIDIA Video Codec SDK（nvEncodeAPI.h），
    不经 FFmpeg 编码器。BGR8 输入 -> BT.709 limited-range NV12（CPU 转换）-> 上传 CUDA device
    buffer -> Register/Map -> nvEncEncodePicture -> LockBitstream 取 H.264 Annex-B 裸流 -> 转换为
    AVCC（剥 AUD/SPS/PPS、改写长度前缀）-> libavformat 仅做 mp4 mux（容器封装，非编码）。
    - CUDA primary context 复用，与 NVDEC 解码链路共存（独立硬件引擎）。
    - 关 B 帧（frameIntervalP=1）/关 lookahead/同步模式，pts=dts=帧计数器单调。
    - SPS/PPS 由 nvEncGetSequenceParams 取出后手工构建 avcC extradata。
    - RAII：析构幂等 close（flush EOS + 写 trailer），异常路径也能保住已编码部分。
*/

#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

struct NvencEncoderImpl;

class NvencEncoder {
public:
    NvencEncoder();
    ~NvencEncoder();

    NvencEncoder(const NvencEncoder&) = delete;
    NvencEncoder& operator=(const NvencEncoder&) = delete;

    // fps>0、宽高均为偶数；失败返回 false（调用方据此降级，不影响 debug 日志/裁剪图）。
    bool open(int width, int height, double fps, const std::string& outPath);
    // BGR8 单帧；失败返回 false。未 open 或已关闭时返回 false。
    bool writeFrame(const cv::Mat& bgr);
    // 幂等：flush EOS + write trailer + 释放全部 NVENC/FFmpeg/CUDA 资源。
    void close();

    bool isOpen() const;

private:
    std::unique_ptr<NvencEncoderImpl> impl_;
};
