/*
Author: wilbur
Version: 1.1
Date: 2026-08-04
Description: NV12(BT.709) → BGR 转换 kernel 与 D2H 下载实现。
    full-range 公式：R=Y+1.5748(V-128)；G=Y-0.1873(U-128)-0.4681(V-128)；B=Y+1.8556(U-128)；
    limited-range：Y'=1.164383(Y-16)，UV 缩放 1.138393 后套同系数。按 frame.fullRange 选择。
    索引一律使用 linesize（不用 width 当行距）；奇数宽高的 UV 采样依赖 NVDEC linesize padding 兜底，
    最后一行/列做边界 min 钳制。
    1.1: 增加 limited-range 分支。
*/

#include "nv12ToBgr.h"

#include "cudaUtils.h"

#include <stdexcept>
#include <string>

namespace {

__global__ void nv12ToBgrKernel(const uint8_t* yPlane, const uint8_t* uvPlane,
                                int width, int height, int yPitch, int uvPitch,
                                uint8_t* bgr, int bgrPitch, int fullRange) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float Y = (float)yPlane[y * yPitch + x];
    // 色度行/列边界钳制（奇数宽高时 (height+1)/2 行 UV）
    int cy = min(y / 2, (height - 1) / 2);
    float U = (float)uvPlane[cy * uvPitch + (x / 2) * 2 + 0];
    float V = (float)uvPlane[cy * uvPitch + (x / 2) * 2 + 1];

    float r, g, b;
    if (fullRange) {
        // BT.709 full-range
        r = Y + 1.5748f * (V - 128.0f);
        g = Y - 0.1873f * (U - 128.0f) - 0.4681f * (V - 128.0f);
        b = Y + 1.8556f * (U - 128.0f);
    } else {
        // BT.709 limited-range：Y'=1.164383(Y-16)，UV 缩放 1.138393
        float ys = 1.164383f * (Y - 16.0f);
        r = ys + 1.79268f * (V - 128.0f);
        g = ys - 0.21322f * (U - 128.0f) - 0.53288f * (V - 128.0f);
        b = ys + 2.11246f * (U - 128.0f);
    }

    uint8_t* out = bgr + y * bgrPitch + x * 3;
    out[0] = (uint8_t)fminf(fmaxf(b, 0.0f), 255.0f);
    out[1] = (uint8_t)fminf(fmaxf(g, 0.0f), 255.0f);
    out[2] = (uint8_t)fminf(fmaxf(r, 0.0f), 255.0f);
}

} // namespace

cv::Mat nv12ToBgr(const GpuFrame& frame, cudaStream_t stream) {
    if (frame.pixFmt != Frame::PixFmt::kNv12 || !frame.data0 || !frame.data1) {
        throw std::runtime_error("nv12ToBgr: frame is not NV12 or has null planes");
    }

    const int width = frame.width;
    const int height = frame.height;
    const size_t bgrBytes = static_cast<size_t>(width) * height * 3;

    // 简单方案：每次 cudaMalloc/cudaFree（D2H 不是瓶颈）
    DeviceBuffer devBgr(bgrBytes);

    dim3 block(32, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    nv12ToBgrKernel<<<grid, block, 0, stream>>>(
        frame.data0, frame.data1, width, height,
        frame.linesize0, frame.linesize1,
        static_cast<uint8_t*>(devBgr.data()), width * 3, frame.fullRange ? 1 : 0);
    checkCuda(cudaGetLastError(), "nv12ToBgrKernel launch");

    cv::Mat bgr(height, width, CV_8UC3);
    checkCuda(cudaMemcpyAsync(bgr.data, devBgr.data(), bgrBytes, cudaMemcpyDeviceToHost, stream),
              "nv12ToBgr D2H copy");
    checkCuda(cudaStreamSynchronize(stream), "nv12ToBgr cudaStreamSynchronize");
    return bgr;
}
