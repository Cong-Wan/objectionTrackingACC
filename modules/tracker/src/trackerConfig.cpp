/*
Author:
Version: 1.5
Date: 2026-06-06
Description: 翻译 config.yaml 的 ByteTrack 追踪参数配置实现，底层改用 yaml-cpp 直接读取 trackers_cpp/config.yaml
*/

#include "trackerConfig.h"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>
#include <vector>

namespace trackerCpp {
namespace {

// 尝试从指定路径加载 YAML 配置文件，成功返回 true，失败返回 false 且不修改 config
bool tryLoadTrackerConfig(const std::string& configPath, TrackerConfig& config) {
    try {
        YAML::Node root = YAML::LoadFile(configPath);
        if (!root || !root.IsMap()) {
            return false;
        }
        // 辅助 lambda：若 YAML 中存在指定 key，则将其转换为对应类型写入目标变量
        auto assignDouble = [&](const char* key, double& dst) {
            if (root[key]) {
                dst = root[key].as<double>();
            }
        };
        auto assignInt = [&](const char* key, int& dst) {
            if (root[key]) {
                dst = root[key].as<int>();
            }
        };
        auto assignBool = [&](const char* key, bool& dst) {
            if (root[key]) {
                dst = root[key].as<bool>();
            }
        };
        assignDouble("track_high_thresh", config.trackHighThresh);
        assignDouble("track_low_thresh", config.trackLowThresh);
        assignDouble("new_track_thresh", config.newTrackThresh);
        assignInt("track_buffer", config.trackBuffer);
        assignDouble("match_thresh", config.matchThresh);
        assignDouble("aspect_ratio_thresh", config.aspectRatioThresh);
        assignDouble("min_box_area", config.minBoxArea);
        assignBool("mot20", config.mot20);
        assignBool("fuse_score", config.fuseScore);
        assignInt("frame_rate", config.frameRate);
        return true;
    } catch (const YAML::BadFile&) {
        return false;
    }
}

} // namespace

// 返回一组硬编码的默认追踪参数
TrackerConfig defaultTrackerConfig() {
    return TrackerConfig{};
}

// 从指定路径加载配置，若文件不存在或解析失败则使用默认值
TrackerConfig loadTrackerConfig(const std::string& configPath) {
    TrackerConfig config = defaultTrackerConfig();
    tryLoadTrackerConfig(configPath, config);
    return config;
}

// 自动推断配置文件路径（源码所在目录的父目录下的 config.yaml）并加载
TrackerConfig loadTrackerConfig() {
    TrackerConfig config = defaultTrackerConfig();
    namespace fs = std::filesystem;
    fs::path sourcePath = fs::path(__FILE__).lexically_normal();
    fs::path configPath = sourcePath.parent_path().parent_path() / "config.yaml";
    tryLoadTrackerConfig(configPath.string(), config);
    return config;
}

} // namespace trackerCpp
