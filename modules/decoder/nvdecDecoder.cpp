/*
Author: wilbur
Version: 1.1
Date: 2026-08-04
Description: FFmpeg CUDA/NVDEC 视频解码实现。使用 device 0 的 primary CUDA context，
    强制输出 AV_PIX_FMT_CUDA，校验 NV12/BT.709（full/limited 量程均支持，写入 GpuFrame.fullRange；
    元数据 unspecified 时按 limited + BT.709 回退并各 warn 一次），并在 EOF 时 flush 解码延迟帧。
*/

#include "nvdecDecoder.h"

#include "logger.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct AvFrameDeleter {
    void operator()(AVFrame* f) const {
        if (f) {
            av_frame_free(&f);
        }
    }
};

std::shared_ptr<AVFrame> allocFrameRef() {
    AVFrame* f = av_frame_alloc();
    if (!f) {
        throw std::runtime_error("av_frame_alloc failed");
    }
    return std::shared_ptr<AVFrame>(f, AvFrameDeleter{});
}

enum AVPixelFormat getHwFormat(AVCodecContext* /*ctx*/, const enum AVPixelFormat* pixFmts) {
    for (const enum AVPixelFormat* p = pixFmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_CUDA) {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}

void logAvError(const std::string& context, int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, buf, sizeof(buf));
    logError(context + ": " + std::string(buf));
}

} // namespace

NvdecDecoder::NvdecDecoder(std::string videoPath, AVBufferRef* sharedHwDeviceCtx)
    : videoPath_(std::move(videoPath)), sharedHwDeviceCtx_(sharedHwDeviceCtx) {}

NvdecDecoder::~NvdecDecoder() {
    if (codecCtx_) {
        avcodec_free_context(&codecCtx_);
    }
    if (formatCtx_) {
        avformat_close_input(&formatCtx_);
    }
    if (hwDeviceCtx_) {
        av_buffer_unref(&hwDeviceCtx_);
    }
}

void NvdecDecoder::open() {
    logDebug("FFmpeg version: " + std::string(av_version_info()));
    logInfo("opening video: " + videoPath_);

    int err = avformat_open_input(&formatCtx_, videoPath_.c_str(), nullptr, nullptr);
    if (err < 0) {
        logAvError("avformat_open_input failed", err);
        throw std::runtime_error("Failed to open video: " + videoPath_);
    }
    err = avformat_find_stream_info(formatCtx_, nullptr);
    if (err < 0) {
        logAvError("avformat_find_stream_info failed", err);
        throw std::runtime_error("Failed to find stream info");
    }

    streamIndex_ = av_find_best_stream(formatCtx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (streamIndex_ < 0) {
        throw std::runtime_error("No video stream found in: " + videoPath_);
    }
    AVStream* stream = formatCtx_->streams[streamIndex_];
    const AVCodecParameters* codecPar = stream->codecpar;

    // 旋转 display matrix 不支持矫正：检测到则告警，按存储方向处理
    if (av_packet_side_data_get(codecPar->coded_side_data, codecPar->nb_coded_side_data,
                                AV_PKT_DATA_DISPLAYMATRIX)) {
        logWarn("rotation metadata ignored: " + videoPath_);
    }

    const AVCodec* codec = avcodec_find_decoder(codecPar->codec_id);
    if (!codec) {
        throw std::runtime_error("Decoder not found for codec id " + std::to_string(codecPar->codec_id));
    }
    logDebug("codec name: " + std::string(codec->name));
    logDebug("stream width=" + std::to_string(codecPar->width) + " height=" + std::to_string(codecPar->height));

    // CUDA hw device context：复用外部共享 context（保证与 GPU 预处理/TRT 同一 CUDA context），
    // 否则自行创建并使用 device 0 的 primary context。
    if (sharedHwDeviceCtx_) {
        hwDeviceCtx_ = av_buffer_ref(sharedHwDeviceCtx_);
        if (!hwDeviceCtx_) {
            throw std::runtime_error("av_buffer_ref(sharedHwDeviceCtx) failed");
        }
    } else {
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "primary_ctx", "1", 0);
        err = av_hwdevice_ctx_create(&hwDeviceCtx_, AV_HWDEVICE_TYPE_CUDA, nullptr, opts, 0);
        av_dict_free(&opts);
        if (err < 0 || !hwDeviceCtx_) {
            logAvError("av_hwdevice_ctx_create(CUDA) failed", err);
            throw std::runtime_error("Failed to create CUDA hw device context");
        }
    }

    codecCtx_ = avcodec_alloc_context3(codec);
    if (!codecCtx_) {
        throw std::runtime_error("avcodec_alloc_context3 failed");
    }
    err = avcodec_parameters_to_context(codecCtx_, codecPar);
    if (err < 0) {
        logAvError("avcodec_parameters_to_context failed", err);
        throw std::runtime_error("Failed to copy codec parameters");
    }

    codecCtx_->hw_device_ctx = av_buffer_ref(hwDeviceCtx_);
    codecCtx_->get_format = getHwFormat;
    codecCtx_->pkt_timebase = stream->time_base;

    err = avcodec_open2(codecCtx_, codec, nullptr);
    if (err < 0) {
        logAvError("avcodec_open2 failed", err);
        throw std::runtime_error("Failed to open decoder");
    }

    logInfo("video opened: width=" + std::to_string(width()) + " height=" + std::to_string(height()) +
            " fps=" + std::to_string(fps()) + " (cuda primary context, device 0)");
}

int NvdecDecoder::width() const {
    return codecCtx_ ? codecCtx_->width : 0;
}

int NvdecDecoder::height() const {
    return codecCtx_ ? codecCtx_->height : 0;
}

double NvdecDecoder::fps() const {
    if (!formatCtx_ || streamIndex_ < 0) {
        return 0.0;
    }
    AVStream* stream = formatCtx_->streams[streamIndex_];
    AVRational r = stream->avg_frame_rate;
    if (r.num == 0 || r.den == 0) {
        r = stream->r_frame_rate;
    }
    if (r.num == 0 || r.den == 0) {
        return 0.0;
    }
    return av_q2d(r);
}

bool NvdecDecoder::readFrame(std::unique_ptr<Frame>& frame) {
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        throw std::runtime_error("av_packet_alloc failed");
    }

    auto freePacket = [](AVPacket* p) { av_packet_free(&p); };
    std::unique_ptr<AVPacket, decltype(freePacket)> packetGuard(packet, freePacket);

    while (true) {
        if (!flushing_) {
            int err = av_read_frame(formatCtx_, packet);
            if (err < 0) {
                if (err == AVERROR_EOF || err == AVERROR(EIO)) {
                    logDebug("decoder reached EOF, flushing delayed frames");
                    flushing_ = true;
                    avcodec_send_packet(codecCtx_, nullptr);
                } else {
                    logAvError("av_read_frame failed", err);
                    throw std::runtime_error("av_read_frame failed");
                }
            } else if (packet->stream_index != streamIndex_) {
                continue; // 忽略非视频流
            } else {
                err = avcodec_send_packet(codecCtx_, packet);
                if (err < 0) {
                    logAvError("avcodec_send_packet failed", err);
                    throw std::runtime_error("avcodec_send_packet failed");
                }
            }
        }

        std::shared_ptr<AVFrame> avframe = allocFrameRef();
        int recv = avcodec_receive_frame(codecCtx_, avframe.get());
        if (recv == AVERROR(EAGAIN) || recv == AVERROR_EOF) {
            if (flushing_) {
                // 已排空
                return false;
            }
            continue; // 需要更多 packet
        } else if (recv < 0) {
            logAvError("avcodec_receive_frame failed", recv);
            throw std::runtime_error("avcodec_receive_frame failed");
        }

        // 只接受 CUDA 硬件帧
        if (avframe->format != AV_PIX_FMT_CUDA) {
            throw std::runtime_error("Decoder output is not AV_PIX_FMT_CUDA (got " +
                                     std::to_string(avframe->format) + ")");
        }
        if (!avframe->hw_frames_ctx) {
            throw std::runtime_error("Decoded frame has no hw_frames_ctx");
        }
        AVHWFramesContext* framesCtx = reinterpret_cast<AVHWFramesContext*>(avframe->hw_frames_ctx->data);
        if (framesCtx->sw_format != AV_PIX_FMT_NV12) {
            throw std::runtime_error("CUDA frame sw_format is not NV12 (got " +
                                     std::string(av_get_pix_fmt_name(framesCtx->sw_format)) + ")");
        }
        // 量程：full(pc) / limited(tv) 均支持；unspecified 按行业惯例回退 limited 并 warn 一次
        bool fullRange;
        if (avframe->color_range == AVCOL_RANGE_JPEG) {
            fullRange = true;
        } else if (avframe->color_range == AVCOL_RANGE_MPEG) {
            fullRange = false;
        } else {  // AVCOL_RANGE_UNSPECIFIED
            if (!warnedColorRange_) {
                logWarn("color range unspecified, assume limited (tv): " + videoPath_);
                warnedColorRange_ = true;
            }
            fullRange = false;
        }
        // 色彩空间：BT.709 或 unspecified（回退 BT.709 并 warn 一次）；明确非 BT.709 标记拒绝
        if (avframe->colorspace == AVCOL_SPC_UNSPECIFIED) {
            if (!warnedColorSpace_) {
                logWarn("color space unspecified, assume BT.709: " + videoPath_);
                warnedColorSpace_ = true;
            }
        } else if (avframe->colorspace != AVCOL_SPC_BT709) {
            throw std::runtime_error("Video color space is not BT.709");
        }

        auto gpu = std::make_unique<GpuFrame>();
        gpu->frameId = nextFrameId_++;
        gpu->width = avframe->width;
        gpu->height = avframe->height;
        gpu->device = Frame::Device::kCuda;
        gpu->pixFmt = Frame::PixFmt::kNv12;
        gpu->fullRange = fullRange;
        gpu->data0 = avframe->data[0];
        gpu->data1 = avframe->data[1];
        gpu->linesize0 = avframe->linesize[0];
        gpu->linesize1 = avframe->linesize[1];
        gpu->frameRef = avframe;

        frame = std::move(gpu);
        return true;
    }
}
