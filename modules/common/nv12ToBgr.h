/*
Author: wilbur
Version: 1.0
Date: 2026-08-04
Description: NV12(full-range BT.709) GPU 帧 → 8bit BGR cv::Mat（含 D2H）。
*/

#pragma once

#include "frameTypes.h"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

cv::Mat nv12ToBgr(const GpuFrame& frame, cudaStream_t stream = nullptr);
