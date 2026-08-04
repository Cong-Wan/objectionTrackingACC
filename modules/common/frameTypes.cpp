/*
Author: wilbur
Version: 1.0
Date: 2026-08-04
Description: GpuFrame 析构定义：在 AVFrame 完整类型可见的翻译单元实例化 shared_ptr 析构。
*/

#include "frameTypes.h"

extern "C" {
#include <libavutil/frame.h>
}

GpuFrame::~GpuFrame() = default;
