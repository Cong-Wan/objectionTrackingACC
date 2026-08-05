/*
Author: wilbur
Version: 1.1
Date: 2026-08-04
Description: CUDA GPU 预处理接口。将 NV12 BT.709 GPU 帧（full/limited 量程自适应）转为 NCHW RGB
    float32（letterbox 640x640，padding 114，/255 归一化），直接写入 TensorRT input device buffer。
    1.1: preprocessNv12ToNchw 增加 fullRange 参数，支持 limited-range 视频。
*/

#pragma once

#include "frameTypes.h"

#include <cuda_runtime.h>

LetterboxParams computeLetterboxParams(int originalWidth, int originalHeight, int inputWidth, int inputHeight);

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
    cudaStream_t stream);
