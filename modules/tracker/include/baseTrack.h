/*
Author:
Version: 1.2
Date: 2026-06-06
Description: 翻译 basetrack.py 的 TrackState 与 BaseTrack 基础追踪结构声明，恢复 activate/update 占位接口以贴近 Python 基类
*/

#ifndef TRACKER_CPP_BASE_TRACK_H
#define TRACKER_CPP_BASE_TRACK_H

#include <any>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace trackerCpp {

// 轨迹状态枚举：新创建 / 正在追踪 / 丢失 / 已移除
enum class TrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

class BaseTrack {
public:
    BaseTrack();
    virtual ~BaseTrack() = default;

    int endFrame() const;

    static int nextId();
    static void resetId();

    virtual void activate();
    virtual void predict();
    virtual void update();

    void markLost();
    void markRemoved();

    int trackId;                           // 全局唯一轨迹 ID
    bool isActivated;                      // 是否已激活（首次关联成功）
    TrackState state;                      // 当前轨迹状态
    std::map<int, std::any> history;       // 历史帧信息
    std::vector<std::any> features;        // 外观特征历史
    std::any currFeature;                  // 当前帧外观特征
    double score;                          // 当前检测置信度
    int startFrame;                        // 轨迹起始帧号
    int frameId;                           // 最后更新帧号
    int timeSinceUpdate;                   // 距离上次更新的帧数
    std::pair<double, double> location;    // 轨迹中心位置

private:
    static int count;
};

} // namespace trackerCpp

#endif // TRACKER_CPP_BASE_TRACK_H
