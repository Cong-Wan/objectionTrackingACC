/*
Author: wilbur
Version: 1.1
Date: 2026-08-04
Description: NV12(BT.709，full/limited 量程自适应) GPU 帧 → 8bit BGR cv::Mat（含 D2H）。
    1.1: 按 GpuFrame.fullRange 选择 full/limited 换算公式。
*/

#pragma once

#include "frameTypes.h"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

cv::Mat nv12ToBgr(const GpuFrame& frame, cudaStream_t stream = nullptr);
