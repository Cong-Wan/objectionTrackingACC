/*
Author: wilbur
Version: 1.1
Date: 2026-08-04
Description: CUDA GPU 预处理实现。NV12 BT.709（full/limited 量程自适应）-> RGB，letterbox bilinear
    resize，padding 114，/255 归一化，输出 NCHW float32。坐标映射使用 half-pixel center，与
    OpenCV INTER_LINEAR 一致。
    1.1: kernel 增加 fullRange 分支，limited 量程按 Y'=1.164383(Y-16)、UV 缩放 1.138393 换算。
*/

#include "gpuPreprocess.h"

#include <cmath>

LetterboxParams computeLetterboxParams(int originalWidth, int originalHeight, int inputWidth, int inputHeight) {
    LetterboxParams p;
    p.originalWidth = originalWidth;
    p.originalHeight = originalHeight;
    p.inputWidth = inputWidth;
    p.inputHeight = inputHeight;
    p.scale = std::min(static_cast<float>(inputWidth) / originalWidth,
                        static_cast<float>(inputHeight) / originalHeight);
    p.resizedWidth = static_cast<int>(std::round(originalWidth * p.scale));
    p.resizedHeight = static_cast<int>(std::round(originalHeight * p.scale));
    p.padX = (inputWidth - p.resizedWidth) / 2;
    p.padY = (inputHeight - p.resizedHeight) / 2;
    return p;
}

namespace {

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// bilinear 采样 NV12 的 Y 平面（按 half-pixel center 反映射到源坐标）。
__device__ __forceinline__ float sampleY(const uint8_t* plane, int srcW, int srcH, int pitch, float srcX, float srcY) {
    float x0 = floorf(srcX);
    float y0 = floorf(srcY);
    float fx = srcX - x0;
    float fy = srcY - y0;
    int x0i = static_cast<int>(x0);
    int y0i = static_cast<int>(y0);
    int x1i = x0i + 1;
    int y1i = y0i + 1;
    x0i = x0i < 0 ? 0 : (x0i > srcW - 1 ? srcW - 1 : x0i);
    x1i = x1i < 0 ? 0 : (x1i > srcW - 1 ? srcW - 1 : x1i);
    y0i = y0i < 0 ? 0 : (y0i > srcH - 1 ? srcH - 1 : y0i);
    y1i = y1i < 0 ? 0 : (y1i > srcH - 1 ? srcH - 1 : y1i);

    float v00 = static_cast<float>(plane[y0i * pitch + x0i]);
    float v01 = static_cast<float>(plane[y0i * pitch + x1i]);
    float v10 = static_cast<float>(plane[y1i * pitch + x0i]);
    float v11 = static_cast<float>(plane[y1i * pitch + x1i]);
    float top = v00 + (v01 - v00) * fx;
    float bot = v10 + (v11 - v10) * fx;
    return top + (bot - top) * fy;
}

// bilinear 采样 NV12 的 UV 平面（UV 分辨率为 Y 的一半，按 Y 坐标的一半取 UV 坐标）。
__device__ __forceinline__ void sampleUv(const uint8_t* plane, int srcW, int srcH, int pitch,
                                         float srcX, float srcY, float& u, float& v) {
    int uvW = srcW / 2;
    int uvH = srcH / 2;
    float ux = srcX * 0.5f;
    float uy = srcY * 0.5f;
    float x0 = floorf(ux);
    float y0 = floorf(uy);
    float fx = ux - x0;
    float fy = uy - y0;
    int x0i = static_cast<int>(x0);
    int y0i = static_cast<int>(y0);
    int x1i = x0i + 1;
    int y1i = y0i + 1;
    x0i = x0i < 0 ? 0 : (x0i > uvW - 1 ? uvW - 1 : x0i);
    x1i = x1i < 0 ? 0 : (x1i > uvW - 1 ? uvW - 1 : x1i);
    y0i = y0i < 0 ? 0 : (y0i > uvH - 1 ? uvH - 1 : y0i);
    y1i = y1i < 0 ? 0 : (y1i > uvH - 1 ? uvH - 1 : y1i);

    // NV12: UV 交错存储，每像素 2 字节 (U, V)
    float u00 = static_cast<float>(plane[y0i * pitch + x0i * 2 + 0]);
    float u01 = static_cast<float>(plane[y0i * pitch + x1i * 2 + 0]);
    float u10 = static_cast<float>(plane[y1i * pitch + x0i * 2 + 0]);
    float u11 = static_cast<float>(plane[y1i * pitch + x1i * 2 + 0]);
    float v00 = static_cast<float>(plane[y0i * pitch + x0i * 2 + 1]);
    float v01 = static_cast<float>(plane[y0i * pitch + x1i * 2 + 1]);
    float v10 = static_cast<float>(plane[y1i * pitch + x0i * 2 + 1]);
    float v11 = static_cast<float>(plane[y1i * pitch + x1i * 2 + 1]);
    float utop = u00 + (u01 - u00) * fx;
    float ubot = u10 + (u11 - u10) * fx;
    float vtop = v00 + (v01 - v00) * fx;
    float vbot = v10 + (v11 - v10) * fx;
    u = utop + (ubot - utop) * fy;
    v = vtop + (vbot - vtop) * fy;
}

__global__ void nv12ToNchwKernel(
    const uint8_t* yPlane,
    const uint8_t* uvPlane,
    int srcWidth,
    int srcHeight,
    int yPitch,
    int uvPitch,
    float* dstNchw,
    int batchIndex,
    int inputWidth,
    int inputHeight,
    int resizedWidth,
    int resizedHeight,
    int padX,
    int padY,
    float scale,
    int fullRange) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= inputWidth || dy >= inputHeight) {
        return;
    }

    int area = inputWidth * inputHeight;
    size_t batchOffset = static_cast<size_t>(batchIndex) * 3 * area;
    float* rPlane = dstNchw + batchOffset + 0 * area;
    float* gPlane = dstNchw + batchOffset + 1 * area;
    float* bPlane = dstNchw + batchOffset + 2 * area;
    int outIdx = dy * inputWidth + dx;

    bool inside = (dx >= padX) && (dx < padX + resizedWidth) &&
                  (dy >= padY) && (dy < padY + resizedHeight);
    if (!inside) {
        // padding 114 -> 114/255
        float pad = 114.0f / 255.0f;
        rPlane[outIdx] = pad;
        gPlane[outIdx] = pad;
        bPlane[outIdx] = pad;
        return;
    }

    // half-pixel center 反映射到源 NV12 坐标
    float dstX = static_cast<float>(dx - padX);
    float dstY = static_cast<float>(dy - padY);
    float srcX = (dstX + 0.5f) / scale - 0.5f;
    float srcY = (dstY + 0.5f) / scale - 0.5f;
    srcX = clampf(srcX, 0.0f, static_cast<float>(srcWidth - 1));
    srcY = clampf(srcY, 0.0f, static_cast<float>(srcHeight - 1));

    float y = sampleY(yPlane, srcWidth, srcHeight, yPitch, srcX, srcY);
    float u, vv;
    sampleUv(uvPlane, srcWidth, srcHeight, uvPitch, srcX, srcY, u, vv);

    float uu = u - 128.0f;
    float wv = vv - 128.0f;
    float r, g, b;
    if (fullRange) {
        // full-range BT.709
        r = y + 1.5748f * wv;
        g = y - 0.1873f * uu - 0.4681f * wv;
        b = y + 1.8556f * uu;
    } else {
        // limited-range BT.709：Y'=1.164383(Y-16)，UV 缩放 1.138393 后套 full 公式系数
        float ys = 1.164383f * (y - 16.0f);
        r = ys + 1.79268f * wv;
        g = ys - 0.21322f * uu - 0.53288f * wv;
        b = ys + 2.11246f * uu;
    }
    r = clampf(r, 0.0f, 255.0f);
    g = clampf(g, 0.0f, 255.0f);
    b = clampf(b, 0.0f, 255.0f);

    rPlane[outIdx] = r / 255.0f;
    gPlane[outIdx] = g / 255.0f;
    bPlane[outIdx] = b / 255.0f;
}

} // namespace

void preprocessNv12ToNchw(
    const uint8_t* yPlane,
    const uint8_t* uvPlane,
    int srcWidth,
    int srcHeight,
    int yPitch,
    int uvPitch,
    float* dstNchw,
    int batchIndex,
    int batchSize,
    int inputWidth,
    int inputHeight,
    const LetterboxParams& params,
    bool fullRange,
    cudaStream_t stream) {
    (void)batchSize; // 仅 batchIndex 决定输出偏移
    dim3 block(16, 16);
    dim3 grid((inputWidth + block.x - 1) / block.x, (inputHeight + block.y - 1) / block.y);
    nv12ToNchwKernel<<<grid, block, 0, stream>>>(
        yPlane, uvPlane, srcWidth, srcHeight, yPitch, uvPitch, dstNchw,
        batchIndex, inputWidth, inputHeight,
        params.resizedWidth, params.resizedHeight, params.padX, params.padY, params.scale,
        fullRange ? 1 : 0);
}
