/*
Author: wilbur
Version: 1.0
Date: 2026-08-04
Description: 推理引擎抽象接口：load/infer，输入 Frame 基类，输出原图坐标系 Detection 列表。
*/

#pragma once

#include "frameTypes.h"

#include <vector>

class IInferEngine {
public:
    virtual ~IInferEngine() = default;
    virtual void load() = 0;
    virtual std::vector<Detection> infer(const Frame& frame) = 0;
};
