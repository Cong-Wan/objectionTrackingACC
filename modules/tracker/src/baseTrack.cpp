/*
Author:
Version: 1.2
Date: 2026-06-06
Description: 翻译 basetrack.py 的 TrackState 与 BaseTrack 基础追踪结构实现，恢复 activate/update 占位接口以贴近 Python 基类
*/

#include "baseTrack.h"

#include <stdexcept>

namespace trackerCpp {

int BaseTrack::count = 0;  // 全局静态轨迹 ID 计数器

BaseTrack::BaseTrack()
    : trackId(0),
      isActivated(false),
      state(TrackState::New),
      score(0.0),
      startFrame(0),
      frameId(0),
      timeSinceUpdate(0),
      location({std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()}) {}

// 返回轨迹最后更新的帧号
int BaseTrack::endFrame() const {
    return frameId;
}

// 自增并返回下一个可用轨迹 ID
int BaseTrack::nextId() {
    count += 1;
    return count;
}

// 重置全局 ID 计数器为 0
void BaseTrack::resetId() {
    count = 0;
}

// 以下三个虚函数为纯占位接口，必须由派生类重写
void BaseTrack::activate() {
    throw std::logic_error("BaseTrack::activate must be implemented by derived classes");
}

void BaseTrack::predict() {
    throw std::logic_error("BaseTrack::predict must be implemented by derived classes");
}

void BaseTrack::update() {
    throw std::logic_error("BaseTrack::update must be implemented by derived classes");
}

// 将轨迹标记为丢失状态
void BaseTrack::markLost() {
    state = TrackState::Lost;
}

// 将轨迹标记为已移除状态
void BaseTrack::markRemoved() {
    state = TrackState::Removed;
}

} // namespace trackerCpp
