/*
Author: wilbur
Version: 1.0
Date: 2026-08-04
Description: 视频解码器抽象接口：open/readFrame/width/height/fps，接口只依赖 Frame 基类。
*/

#pragma once

#include "frameTypes.h"

#include <memory>

class IVideoDecoder {
public:
    virtual ~IVideoDecoder() = default;
    virtual void open() = 0;
    virtual bool readFrame(std::unique_ptr<Frame>& frame) = 0;  // false = 正常结束（EOF 已排空）
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual double fps() const = 0;
};
