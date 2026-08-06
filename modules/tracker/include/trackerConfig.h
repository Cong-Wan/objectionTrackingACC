/*
Author: wilbur
Version: 1.3
Date: 2026-08-06
Description: 翻译 config.yaml 的 ByteTrack 追踪参数配置声明，支持从配置文件读取并用默认值兜底；底层改用 yaml-cpp 解析
    1.3: track_buffer / frame_rate 注释改为按 tracker 帧语义（maxTimeLost=max(1,buffer)，与跳帧 N 无关计数）。
*/

#ifndef TRACKER_CPP_TRACKER_CONFIG_H
#define TRACKER_CPP_TRACKER_CONFIG_H

#include <string>

namespace trackerCpp {

struct TrackerConfig {
    double trackHighThresh = 0.5;     // 高分检测阈值，>= 此值的检测框用于第一级匹配
    double trackLowThresh = 0.05;     // 低分检测阈值，介于低分与高分之间的检测框用于第二级匹配
    double newTrackThresh = 0.5;      // 新建追踪轨迹的置信度阈值
    int trackBuffer = 5;              // 丢失后保留的 tracker 处理帧数；maxTimeLost=max(1,trackBuffer)
    double matchThresh = 0.9;         // 匈牙利匹配时的代价矩阵阈值
    double aspectRatioThresh = 1.5;   // 宽高比阈值，用于过滤异常检测框
    double minBoxArea = 10.0;         // 最小检测框面积阈值
    bool mot20 = false;               // 是否启用 MOT20 模式
    bool fuseScore = false;           // 是否将检测置信度融合进 IoU 距离
    int frameRate = 30;               // 有效处理帧率（应用层常设为 video_fps/N，仅日志/换算秒数）
};

TrackerConfig defaultTrackerConfig();
TrackerConfig loadTrackerConfig();
TrackerConfig loadTrackerConfig(const std::string& configPath);

} // namespace trackerCpp

#endif // TRACKER_CPP_TRACKER_CONFIG_H
