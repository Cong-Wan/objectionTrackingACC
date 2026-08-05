/*
Author: wilbur
Version: 1.0
Date: 2026-08-05
Description: NVENC 硬编码视频编码器实现。详见 nvencEncoder.h。
*/

#include "nvencEncoder.h"

#include "logger.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/rational.h>
#include <libavutil/frame.h>
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <ffnvcodec/nvEncodeAPI.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ===== FFmpeg 错误日志 =====
void logAvErrorNv(const std::string& context, int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, buf, sizeof(buf));
    logError(context + ": " + std::string(buf));
}

// ===== Annex-B -> AVCC 转换 =====
// 解析 Annex-B（start code 分隔），改写为 AVCC（4 字节大端长度前缀），剥除 AUD(9)/SPS(7)/PPS(8)。
// 返回转换后的 AVCC 字节流；badData=true 表示无可用 start code。
std::vector<uint8_t> annexbToAvcc(const uint8_t* data, size_t size, bool& badData) {
    badData = false;
    std::vector<uint8_t> out;
    if (size < 4) return out;
    size_t i = 0;
    auto findNextStartCode = [&](size_t from) -> size_t {
        for (size_t p = from; p + 3 < size; ++p) {
            if (data[p] == 0 && data[p + 1] == 0) {
                if (data[p + 2] == 1) return p;            // 00 00 01
                if (data[p + 2] == 0 && p + 3 < size && data[p + 3] == 1) return p;  // 00 00 00 01
            }
        }
        return size;
    };
    while (i < size) {
        size_t sc = findNextStartCode(i);
        if (sc == size) break;
        // 跳过 start code（3 或 4 字节）
        size_t hdr = (data[sc + 2] == 1) ? 3 : 4;
        size_t nalStart = sc + hdr;
        // 找下一个 start code 或结尾
        size_t next = findNextStartCode(nalStart);
        if (next > size) next = size;
        size_t nalLen = next - nalStart;
        if (nalLen == 0) { i = next; continue; }
        uint8_t nalType = data[nalStart] & 0x1F;
        // mp4 包内只保留 VCL 与 SEI，剥除 AUD/SPS/PPS（SPS/PPS 已进 extradata 且 repeatSPSPPS 会重复）
        if (nalType != 9 && nalType != 7 && nalType != 8) {
            // 4 字节大端长度前缀
            uint32_t len = static_cast<uint32_t>(nalLen);
            out.push_back(static_cast<uint8_t>((len >> 24) & 0xFF));
            out.push_back(static_cast<uint8_t>((len >> 16) & 0xFF));
            out.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
            out.push_back(static_cast<uint8_t>(len & 0xFF));
            out.insert(out.end(), data + nalStart, data + nalStart + nalLen);
        }
        i = next;
    }
    if (out.empty()) badData = true;
    return out;
}

// 从 Annex-B 序列参数 payload 中解析出第一个 SPS 与第一个 PPS 的 NAL 负载（含 NAL header 字节）。
// 返回的 pair.first=SPS，pair.second=PPS；任一为空表示未找到。
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> extractSpsPps(const uint8_t* data, size_t size) {
    std::vector<uint8_t> sps, pps;
    size_t i = 0;
    auto findNextStartCode = [&](size_t from) -> size_t {
        for (size_t p = from; p + 3 < size; ++p) {
            if (data[p] == 0 && data[p + 1] == 0) {
                if (data[p + 2] == 1) return p;
                if (data[p + 2] == 0 && p + 3 < size && data[p + 3] == 1) return p;
            }
        }
        return size;
    };
    while (i < size) {
        size_t sc = findNextStartCode(i);
        if (sc == size) break;
        size_t hdr = (data[sc + 2] == 1) ? 3 : 4;
        size_t nalStart = sc + hdr;
        size_t next = findNextStartCode(nalStart);
        if (next > size) next = size;
        size_t nalLen = next - nalStart;
        if (nalLen > 0) {
            uint8_t nalType = data[nalStart] & 0x1F;
            if (nalType == 7 && sps.empty()) {
                sps.assign(data + nalStart, data + nalStart + nalLen);
            } else if (nalType == 8 && pps.empty()) {
                pps.assign(data + nalStart, data + nalStart + nalLen);
            }
        }
        i = next;
    }
    return {sps, pps};
}

// 由 SPS/PPS 构建 AVCDecoderConfigurationRecord（avcC）。
// [0]=1, [1..3]=profile/compat/level(取自 SPS 负载), [4]=0xFF(NALU length 4),
// [5]=0xE0|numSPS, uint16 SPS len + SPS, [6]=numPPS, uint16 PPS len + PPS。
std::vector<uint8_t> buildAvcC(const std::vector<uint8_t>& sps, const std::vector<uint8_t>& pps) {
    std::vector<uint8_t> avcC;
    if (sps.size() < 4 || pps.empty()) return avcC;  // SPS 至少需 header + 3 字节
    avcC.push_back(1);                               // version
    avcC.push_back(sps[1]);                           // profile_idc
    avcC.push_back(sps[2]);                           // constraint flags
    avcC.push_back(sps[3]);                           // level_idc
    avcC.push_back(0xFF);                             // 6 reserved bits + lengthSizeMinusOne=3
    avcC.push_back(static_cast<uint8_t>(0xE0 | 1));   // 3 reserved bits + numSPS=1
    avcC.push_back(static_cast<uint8_t>((sps.size() >> 8) & 0xFF));
    avcC.push_back(static_cast<uint8_t>(sps.size() & 0xFF));
    avcC.insert(avcC.end(), sps.begin(), sps.end());
    avcC.push_back(1);                                // numPPS=1
    avcC.push_back(static_cast<uint8_t>((pps.size() >> 8) & 0xFF));
    avcC.push_back(static_cast<uint8_t>(pps.size() & 0xFF));
    avcC.insert(avcC.end(), pps.begin(), pps.end());
    return avcC;
}

// ===== BGR -> NV12 (BT.709 limited-range) CPU 转换 =====
// Y = 16 + 0.18265*R + 0.61424*G + 0.06200*B
// U = 128 - 0.10064*R - 0.33852*G + 0.43916*B
// V = 128 + 0.43908*R - 0.39889*G - 0.04027*B
void bgrToNv12Bt709Limited(const cv::Mat& bgr, std::vector<uint8_t>& nv12) {
    const int w = bgr.cols;
    const int h = bgr.rows;
    const size_t ySize = static_cast<size_t>(w) * h;
    nv12.resize(ySize * 3 / 2);
    uint8_t* yPlane = nv12.data();
    uint8_t* uvPlane = yPlane + ySize;
    const int uvH = h / 2;
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = bgr.ptr<uint8_t>(y);
        uint8_t* yRow = yPlane + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const uint8_t* px = row + x * 3;
            float b = px[0], g = px[1], r = px[2];
            float yy = 16.0f + 0.18265f * r + 0.61424f * g + 0.06200f * b;
            yRow[x] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, yy)));
        }
    }
    for (int y = 0; y < uvH; ++y) {
        const uint8_t* row0 = bgr.ptr<uint8_t>(y * 2);
        const uint8_t* row1 = bgr.ptr<uint8_t>(y * 2 + 1);
        uint8_t* uvRow = uvPlane + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; x += 2) {
            const uint8_t* px0 = row0 + x * 3;
            const uint8_t* px1 = row0 + (x + 1) * 3;
            const uint8_t* px2 = row1 + x * 3;
            const uint8_t* px3 = row1 + (x + 1) * 3;
            float b = (px0[0] + px1[0] + px2[0] + px3[0]) * 0.25f;
            float g = (px0[1] + px1[1] + px2[1] + px3[1]) * 0.25f;
            float r = (px0[2] + px1[2] + px2[2] + px3[2]) * 0.25f;
            float u = 128.0f - 0.10064f * r - 0.33852f * g + 0.43916f * b;
            float v = 128.0f + 0.43908f * r - 0.39889f * g - 0.04027f * b;
            uvRow[x] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, u)));
            uvRow[x + 1] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, v)));
        }
    }
}

} // namespace

// ===== Impl =====
struct NvencEncoderImpl {
    // NVENC
    NV_ENCODE_API_FUNCTION_LIST fnList{};
    void* encoder = nullptr;
    NV_ENC_REGISTER_RESOURCE regRes{};
    bool registered = false;
    NV_ENC_CREATE_BITSTREAM_BUFFER bitBuf{};
    bool bitBufCreated = false;
    // CUDA 输入 buffer
    uint8_t* devNv12 = nullptr;
    size_t devPitch = 0;
    CUcontext cuCtx = nullptr;  // 编码器绑定的 CUDA context，写帧时需 push 为当前
    int width = 0;
    int height = 0;
    AVRational frameRate{};
    AVRational timeBase{};
    // mux
    AVFormatContext* fmtCtx = nullptr;
    AVStream* stream = nullptr;
    int64_t frameIdx = 0;
    bool opened = false;
    bool broken = false;  // 编码/写帧失败后置位，后续 writeFrame 直接返回 false

    // 临时缓冲
    std::vector<uint8_t> nv12Buf;
    std::vector<uint8_t> bitstreamBuf;  // 用于 GetSequenceParams 与每帧 bitstream 复制

    ~NvencEncoderImpl() { cleanup(); }

    void cleanup() {
        if (opened) flushAndDestroyEncoder();
        destroyMux();
        if (devNv12) {
            cudaFree(devNv12);
            devNv12 = nullptr;
        }
        opened = false;
    }

    void flushAndDestroyEncoder() {
        if (!encoder) return;
        // EOS flush：frameIntervalP=1 + 同步模式无延迟帧；EOS 只发一次，再 lock 取残余包（不循环 Encode）
        if (bitBufCreated && !broken) {
            NV_ENC_PIC_PARAMS pic{};
            pic.version = NV_ENC_PIC_PARAMS_VER;
            pic.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
            pic.inputBuffer = nullptr;
            pic.outputBitstream = bitBuf.bitstreamBuffer;
            NVENCSTATUS st = fnList.nvEncEncodePicture(encoder, &pic);
            if (st == NV_ENC_SUCCESS || st == NV_ENC_ERR_NEED_MORE_INPUT) {
                NV_ENC_LOCK_BITSTREAM lock{};
                lock.version = NV_ENC_LOCK_BITSTREAM_VER;
                lock.outputBitstream = bitBuf.bitstreamBuffer;
                lock.doNotWait = 0;
                if (fnList.nvEncLockBitstream(encoder, &lock) == NV_ENC_SUCCESS) {
                    if (lock.bitstreamSizeInBytes > 0) {
                        bool bad = false;
                        auto avcc = annexbToAvcc(static_cast<const uint8_t*>(lock.bitstreamBufferPtr),
                                                 lock.bitstreamSizeInBytes, bad);
                        if (!bad && !avcc.empty()) {
                            writeMuxPacket(avcc.data(), avcc.size(), frameIdx,
                                           lock.pictureType == NV_ENC_PIC_TYPE_IDR);
                            ++frameIdx;
                        }
                    }
                    fnList.nvEncUnlockBitstream(encoder, bitBuf.bitstreamBuffer);
                }
            }
        }
        if (bitBufCreated) {
            fnList.nvEncDestroyBitstreamBuffer(encoder, bitBuf.bitstreamBuffer);
            bitBufCreated = false;
        }
        if (registered) {
            fnList.nvEncUnregisterResource(encoder, regRes.registeredResource);
            registered = false;
        }
        fnList.nvEncDestroyEncoder(encoder);
        encoder = nullptr;
    }

    void destroyMux() {
        if (fmtCtx) {
            if (fmtCtx->pb) {
                if (opened && stream) av_write_trailer(fmtCtx);
                avio_closep(&fmtCtx->pb);
            }
            avformat_free_context(fmtCtx);
            fmtCtx = nullptr;
            stream = nullptr;
        }
    }

    bool writeMuxPacket(const uint8_t* data, size_t size, int64_t frameIndex, bool isKey) {
        if (!stream) return false;
        AVPacket* pkt = av_packet_alloc();
        if (!pkt) return false;
        av_new_packet(pkt, static_cast<int>(size));
        std::memcpy(pkt->data, data, size);
        // write_header 后 stream->time_base 可能被 muxer 改写，按 frameRate 时间基重缩放
        AVRational srcTb = av_inv_q(frameRate);
        pkt->pts = av_rescale_q(frameIndex, srcTb, stream->time_base);
        pkt->dts = pkt->pts;
        pkt->duration = av_rescale_q(1, srcTb, stream->time_base);
        pkt->stream_index = stream->index;
        if (isKey) pkt->flags |= AV_PKT_FLAG_KEY;
        int ret = av_interleaved_write_frame(fmtCtx, pkt);
        av_packet_free(&pkt);
        if (ret < 0) {
            logAvErrorNv("nvencEncoder mux write_frame failed", ret);
            return false;
        }
        return true;
    }
};

// ===== 对外接口 =====
NvencEncoder::NvencEncoder() : impl_(std::make_unique<NvencEncoderImpl>()) {}
NvencEncoder::~NvencEncoder() = default;

bool NvencEncoder::isOpen() const { return impl_ && impl_->opened && !impl_->broken; }

bool NvencEncoder::open(int width, int height, double fps, const std::string& outPath) {
    auto& p = *impl_;
    if (p.opened) return false;
    if (width <= 0 || height <= 0 || fps <= 0.0) {
        logError("nvencEncoder open: invalid params");
        return false;
    }
    if ((width & 1) || (height & 1)) {
        logError("nvencEncoder open: width/height must be even (NV12/h264 requirement), got " +
                 std::to_string(width) + "x" + std::to_string(height));
        return false;
    }
    p.width = width;
    p.height = height;
    p.frameRate = av_d2q(fps, 1001000);
    p.timeBase = av_inv_q(p.frameRate);

    // 1) 驱动版本校验
    uint32_t maxVer = 0;
    if (NvEncodeAPIGetMaxSupportedVersion(&maxVer) != NV_ENC_SUCCESS) {
        logError("nvencEncoder: NvEncodeAPIGetMaxSupportedVersion failed");
        return false;
    }
    if (maxVer < NVENCAPI_VERSION) {
        uint32_t drvMaj = maxVer & 0xFF;
        uint32_t drvMin = (maxVer >> 24) & 0xFF;
        uint32_t hdrMaj = NVENCAPI_VERSION & 0xFF;
        uint32_t hdrMin = (NVENCAPI_VERSION >> 24) & 0xFF;
        logError("nvencEncoder: driver NVENC " + std::to_string(drvMaj) + "." +
                 std::to_string(drvMin) + " < header " + std::to_string(hdrMaj) + "." +
                 std::to_string(hdrMin) + ", skip debug video");
        return false;
    }

    // 2) 函数表
    p.fnList.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    NVENCSTATUS st = NvEncodeAPICreateInstance(&p.fnList);
    if (st != NV_ENC_SUCCESS) {
        logError("nvencEncoder: NvEncodeAPICreateInstance failed, code=" + std::to_string(st));
        return false;
    }

    // 3) 确保 CUDA primary context 就绪并取出（与 NVDEC/TRT 共用同一 primary context）
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(0) != cudaSuccess) {
        logError("nvencEncoder: cudaSetDevice/cudaFree init primary context failed");
        return false;
    }
    CUcontext cuCtx = nullptr;
    if (cuCtxGetCurrent(&cuCtx) != CUDA_SUCCESS || !cuCtx) {
        logError("nvencEncoder: cuCtxGetCurrent failed");
        return false;
    }
    p.cuCtx = cuCtx;

    // 4) OpenEncodeSessionEx
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sess{};
    sess.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    sess.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    sess.device = cuCtx;
    sess.apiVersion = NVENCAPI_VERSION;
    st = p.fnList.nvEncOpenEncodeSessionEx(&sess, &p.encoder);
    if (st != NV_ENC_SUCCESS || !p.encoder) {
        logError("nvencEncoder: nvEncOpenEncodeSessionEx failed, code=" + std::to_string(st));
        return false;
    }

    // 5) 取 P4 preset config 并修改（用 Ex 接口 + HIGH_QUALITY tuning；再覆盖关 B 帧/lookahead）
    NV_ENC_PRESET_CONFIG presetCfg{};
    presetCfg.version = NV_ENC_PRESET_CONFIG_VER;
    presetCfg.presetCfg.version = NV_ENC_CONFIG_VER;
    NV_ENC_TUNING_INFO tuning = NV_ENC_TUNING_INFO_HIGH_QUALITY;
    st = p.fnList.nvEncGetEncodePresetConfigEx(p.encoder, NV_ENC_CODEC_H264_GUID,
                                               NV_ENC_PRESET_P4_GUID, tuning, &presetCfg);
    if (st != NV_ENC_SUCCESS) {
        // 回退到不带 tuning 的旧接口
        tuning = NV_ENC_TUNING_INFO_UNDEFINED;
        st = p.fnList.nvEncGetEncodePresetConfig(p.encoder, NV_ENC_CODEC_H264_GUID,
                                                 NV_ENC_PRESET_P4_GUID, &presetCfg);
    }
    if (st != NV_ENC_SUCCESS) {
        logError("nvencEncoder: nvEncGetEncodePresetConfig failed, code=" + std::to_string(st));
        p.fnList.nvEncDestroyEncoder(p.encoder);
        p.encoder = nullptr;
        return false;
    }
    NV_ENC_CONFIG& cfg = presetCfg.presetCfg;
    cfg.gopLength = static_cast<uint32_t>(std::max(1, static_cast<int>(2.0 * fps)));
    cfg.frameIntervalP = 1;  // 关 B 帧，pts 单调
    cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
    uint32_t avg = static_cast<uint32_t>(static_cast<double>(width) * height * fps * 0.07);
    avg = std::max<uint32_t>(1000000, std::min<uint32_t>(avg, 20000000));  // 1~20Mbps
    cfg.rcParams.averageBitRate = avg;
    cfg.rcParams.maxBitRate = avg + avg / 2;
    cfg.rcParams.vbvBufferSize = 0;
    cfg.rcParams.enableLookahead = 0;

    NV_ENC_CONFIG_H264& h264 = cfg.encodeCodecConfig.h264Config;
    h264.repeatSPSPPS = 1;
    // VUI：BT.709 limited-range
    NV_ENC_CONFIG_H264_VUI_PARAMETERS& vui = h264.h264VUIParameters;
    vui.videoSignalTypePresentFlag = 1;
    vui.videoFullRangeFlag = 0;  // limited range
    vui.colourDescriptionPresentFlag = 1;
    vui.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_BT709;
    vui.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
    vui.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS_BT709;

    // 6) InitializeEncoder
    NV_ENC_INITIALIZE_PARAMS initParams{};
    initParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
    initParams.encodeGUID = NV_ENC_CODEC_H264_GUID;
    initParams.presetGUID = NV_ENC_PRESET_P4_GUID;
    initParams.encodeWidth = static_cast<uint32_t>(width);
    initParams.encodeHeight = static_cast<uint32_t>(height);
    initParams.darWidth = static_cast<uint32_t>(width);
    initParams.darHeight = static_cast<uint32_t>(height);
    initParams.frameRateNum = static_cast<uint32_t>(p.frameRate.num);
    initParams.frameRateDen = static_cast<uint32_t>(p.frameRate.den);
    initParams.enableEncodeAsync = 0;
    initParams.enablePTD = 1;
    initParams.tuningInfo = tuning;
    initParams.encodeConfig = &cfg;
    st = p.fnList.nvEncInitializeEncoder(p.encoder, &initParams);
    if (st != NV_ENC_SUCCESS) {
        logError("nvencEncoder: nvEncInitializeEncoder failed, code=" + std::to_string(st));
        p.fnList.nvEncDestroyEncoder(p.encoder);
        p.encoder = nullptr;
        return false;
    }

    // 7) GetSequenceParams -> SPS/PPS -> avcC（仅 2 参数：encoder + payload；outSPSPPSPayloadSize 回写实际长度）
    const size_t seqBufCap = 4096;
    p.bitstreamBuf.resize(seqBufCap);
    uint32_t spsPpsSize = 0;
    NV_ENC_SEQUENCE_PARAM_PAYLOAD seq{};
    seq.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
    seq.inBufferSize = static_cast<uint32_t>(seqBufCap);
    seq.spsppsBuffer = p.bitstreamBuf.data();
    seq.outSPSPPSPayloadSize = &spsPpsSize;
    st = p.fnList.nvEncGetSequenceParams(p.encoder, &seq);
    if (st != NV_ENC_SUCCESS) {
        logError("nvencEncoder: nvEncGetSequenceParams failed, code=" + std::to_string(st));
        p.fnList.nvEncDestroyEncoder(p.encoder);
        p.encoder = nullptr;
        return false;
    }
    auto [sps, pps] = extractSpsPps(p.bitstreamBuf.data(), spsPpsSize);
    std::vector<uint8_t> avcC = buildAvcC(sps, pps);
    if (avcC.empty()) {
        logError("nvencEncoder: build avcC failed (SPS/PPS not found)");
        p.fnList.nvEncDestroyEncoder(p.encoder);
        p.encoder = nullptr;
        return false;
    }

    // 8) CreateBitstreamBuffer
    p.bitBuf.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    p.bitBuf.size = static_cast<uint32_t>(seqBufCap);
    st = p.fnList.nvEncCreateBitstreamBuffer(p.encoder, &p.bitBuf);
    if (st != NV_ENC_SUCCESS) {
        logError("nvencEncoder: nvEncCreateBitstreamBuffer failed, code=" + std::to_string(st));
        p.fnList.nvEncDestroyEncoder(p.encoder);
        p.encoder = nullptr;
        return false;
    }
    p.bitBufCreated = true;

    // 9) 分配 CUDA NV12 device buffer 并注册
    cudaError_t cerr = cudaMallocPitch(reinterpret_cast<void**>(&p.devNv12), &p.devPitch,
                                       static_cast<size_t>(width),
                                       static_cast<size_t>(height * 3 / 2));
    if (cerr != cudaSuccess || !p.devNv12) {
        logError("nvencEncoder: cudaMallocPitch failed: " + std::string(cudaGetErrorString(cerr)));
        p.fnList.nvEncDestroyBitstreamBuffer(p.encoder, p.bitBuf.bitstreamBuffer);
        p.fnList.nvEncDestroyEncoder(p.encoder);
        p.encoder = nullptr;
        return false;
    }
    if (p.devPitch % 4 != 0) {
        // NVENC 要求 pitch 是 4 的倍数；cudaMallocPitch 通常已对齐
        logWarn("nvencEncoder: pitch " + std::to_string(p.devPitch) + " not multiple of 4");
    }

    p.regRes.version = NV_ENC_REGISTER_RESOURCE_VER;
    p.regRes.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    p.regRes.resourceToRegister = p.devNv12;
    p.regRes.width = static_cast<uint32_t>(width);
    p.regRes.height = static_cast<uint32_t>(height);
    p.regRes.pitch = static_cast<uint32_t>(p.devPitch);
    p.regRes.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
    p.regRes.bufferUsage = NV_ENC_INPUT_IMAGE;
    st = p.fnList.nvEncRegisterResource(p.encoder, &p.regRes);
    if (st != NV_ENC_SUCCESS || !p.regRes.registeredResource) {
        logError("nvencEncoder: nvEncRegisterResource failed, code=" + std::to_string(st) +
                 " registered=" + std::to_string((uintptr_t)p.regRes.registeredResource));
        cudaFree(p.devNv12);
        p.devNv12 = nullptr;
        p.fnList.nvEncDestroyBitstreamBuffer(p.encoder, p.bitBuf.bitstreamBuffer);
        p.fnList.nvEncDestroyEncoder(p.encoder);
        p.encoder = nullptr;
        return false;
    }
    p.registered = true;
    logInfo("nvencEncoder: registered NV12 device buffer (ctx=" +
            std::to_string((uintptr_t)cuCtx) + ", devPtr=" + std::to_string((uintptr_t)p.devNv12) +
            ", pitch=" + std::to_string(p.devPitch) + ")");

    // 10) mp4 mux
    int ret = avformat_alloc_output_context2(&p.fmtCtx, nullptr, "mp4", outPath.c_str());
    if (ret < 0 || !p.fmtCtx) {
        logAvErrorNv("nvencEncoder: avformat_alloc_output_context2 failed", ret);
        p.cleanup();
        return false;
    }
    p.stream = avformat_new_stream(p.fmtCtx, nullptr);
    if (!p.stream) {
        logError("nvencEncoder: avformat_new_stream failed");
        p.cleanup();
        return false;
    }
    AVCodecParameters* par = p.stream->codecpar;
    par->codec_type = AVMEDIA_TYPE_VIDEO;
    par->codec_id = AV_CODEC_ID_H264;
    par->width = width;
    par->height = height;
    par->format = AV_PIX_FMT_YUV420P;
    par->codec_tag = 0;
    par->color_primaries = AVCOL_PRI_BT709;
    par->color_trc = AVCOL_TRC_BT709;
    par->color_space = AVCOL_SPC_BT709;
    par->color_range = AVCOL_RANGE_MPEG;  // limited
    par->bit_rate = avg;
    par->extradata = static_cast<uint8_t*>(av_mallocz(avcC.size() + AV_INPUT_BUFFER_PADDING_SIZE));
    if (!par->extradata) {
        logError("nvencEncoder: av_mallocz extradata failed");
        p.cleanup();
        return false;
    }
    std::memcpy(par->extradata, avcC.data(), avcC.size());
    par->extradata_size = static_cast<int>(avcC.size());
    p.stream->time_base = p.timeBase;
    p.stream->avg_frame_rate = p.frameRate;
    p.stream->r_frame_rate = p.frameRate;

    if (!(p.fmtCtx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&p.fmtCtx->pb, outPath.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            logAvErrorNv("nvencEncoder: avio_open failed: " + outPath, ret);
            p.cleanup();
            return false;
        }
    }
    ret = avformat_write_header(p.fmtCtx, nullptr);
    if (ret < 0) {
        logAvErrorNv("nvencEncoder: avformat_write_header failed", ret);
        p.cleanup();
        return false;
    }

    p.opened = true;
    p.broken = false;
    p.frameIdx = 0;
    logInfo("nvencEncoder opened: " + std::to_string(width) + "x" + std::to_string(height) + "@" +
            std::to_string(fps) + "fps -> " + outPath +
            " (VBR avg=" + std::to_string(avg / 1000) + "kbps, preset P4, GOP=" +
            std::to_string(cfg.gopLength) + ")");
    return true;
}

bool NvencEncoder::writeFrame(const cv::Mat& bgr) {
    auto& p = *impl_;
    if (!p.opened || p.broken) return false;
    if (bgr.empty() || bgr.cols != p.width || bgr.rows != p.height || bgr.type() != CV_8UC3) {
        logError("nvencEncoder writeFrame: bgr size/type mismatch, expect " +
                 std::to_string(p.width) + "x" + std::to_string(p.height) + " CV_8UC3");
        return false;
    }

    // 1) BGR -> NV12 (CPU, BT.709 limited)
    bgrToNv12Bt709Limited(bgr, p.nv12Buf);

    // 确保当前 CUDA context 为编码器会话绑定的 context（解码/TRT 可能切换了当前 ctx）
    CUcontext curCtx = nullptr;
    cuCtxGetCurrent(&curCtx);
    bool pushed = false;
    if (curCtx != p.cuCtx) {
        if (cuCtxPushCurrent(p.cuCtx) == CUDA_SUCCESS) {
            pushed = true;
        } else {
            logError("nvencEncoder writeFrame: cuCtxPushCurrent failed");
            p.broken = true;
            return false;
        }
    }
    struct CtxGuard {
        bool& pushed;
        CUcontext ctx;
        ~CtxGuard() { if (pushed) { CUcontext dummy = nullptr; cuCtxPopCurrent(&dummy); } }
    } guard{pushed, p.cuCtx};
    (void)guard;

    // 2) 上传到 device
    cudaError_t cerr = cudaMemcpy2D(p.devNv12, p.devPitch, p.nv12Buf.data(),
                                   static_cast<size_t>(p.width), p.width, p.height, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        logError("nvencEncoder writeFrame: cudaMemcpy2D Y failed: " + std::string(cudaGetErrorString(cerr)));
        p.broken = true;
        return false;
    }
    // UV 平面紧接 Y 之后
    cerr = cudaMemcpy2D(p.devNv12 + p.devPitch * p.height, p.devPitch,
                       p.nv12Buf.data() + static_cast<size_t>(p.width) * p.height,
                       static_cast<size_t>(p.width), p.width, p.height / 2, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        logError("nvencEncoder writeFrame: cudaMemcpy2D UV failed: " + std::string(cudaGetErrorString(cerr)));
        p.broken = true;
        return false;
    }

    // 3) Map -> Encode -> Unmap
    NV_ENC_MAP_INPUT_RESOURCE map{};
    map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
    // 新版 API 用 registeredResource 字段（inputResource 已废弃）
    map.registeredResource = p.regRes.registeredResource;
    NVENCSTATUS st = p.fnList.nvEncMapInputResource(p.encoder, &map);
    if (st != NV_ENC_SUCCESS) {
        logError("nvencEncoder: nvEncMapInputResource failed, code=" + std::to_string(st));
        p.broken = true;
        return false;
    }

    NV_ENC_PIC_PARAMS pic{};
    pic.version = NV_ENC_PIC_PARAMS_VER;
    pic.inputWidth = static_cast<uint32_t>(p.width);
    pic.inputHeight = static_cast<uint32_t>(p.height);
    pic.inputPitch = static_cast<uint32_t>(p.devPitch);
    pic.inputBuffer = map.mappedResource;
    pic.bufferFmt = map.mappedBufferFmt;
    pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pic.inputTimeStamp = static_cast<uint64_t>(p.frameIdx);
    pic.outputBitstream = p.bitBuf.bitstreamBuffer;
    if (p.frameIdx == 0) pic.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;  // 首帧强制 IDR
    st = p.fnList.nvEncEncodePicture(p.encoder, &pic);
    // Unmap 无论 encode 是否成功都要做
    p.fnList.nvEncUnmapInputResource(p.encoder, map.mappedResource);
    if (st != NV_ENC_SUCCESS) {
        logError("nvencEncoder: nvEncEncodePicture failed, code=" + std::to_string(st));
        p.broken = true;
        return false;
    }

    // 4) LockBitstream 取流
    NV_ENC_LOCK_BITSTREAM lock{};
    lock.version = NV_ENC_LOCK_BITSTREAM_VER;
    lock.outputBitstream = p.bitBuf.bitstreamBuffer;
    lock.doNotWait = 0;
    st = p.fnList.nvEncLockBitstream(p.encoder, &lock);
    if (st != NV_ENC_SUCCESS) {
        logError("nvencEncoder: nvEncLockBitstream failed, code=" + std::to_string(st));
        p.broken = true;
        return false;
    }
    bool isKey = (lock.pictureType == NV_ENC_PIC_TYPE_IDR);
    auto avcc = annexbToAvcc(static_cast<const uint8_t*>(lock.bitstreamBufferPtr),
                             lock.bitstreamSizeInBytes, p.broken);
    p.fnList.nvEncUnlockBitstream(p.encoder, p.bitBuf.bitstreamBuffer);
    if (p.broken || avcc.empty()) {
        logError("nvencEncoder: empty/invalid bitstream at frame " + std::to_string(p.frameIdx));
        p.broken = true;
        return false;
    }

    // 5) mux
    bool ok = p.writeMuxPacket(avcc.data(), avcc.size(), p.frameIdx, isKey);
    ++p.frameIdx;
    if (!ok) {
        p.broken = true;
        return false;
    }
    return true;
}

void NvencEncoder::close() {
    if (impl_) impl_->cleanup();
}
