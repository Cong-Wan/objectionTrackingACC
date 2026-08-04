/*
Author: wilbur
Version: 1.0
Date: 2026-06-27
Description: CUDA GPU 预处理接口。将 NV12 full-range BT.709 GPU 帧转为 NCHW RGB float32
    （letterbox 640x640，padding 114，/255 归一化），直接写入 TensorRT input device buffer。
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
    cudaStream_t stream);
