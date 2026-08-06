/*
Author: wilbur
Version: 1.6
Date: 2026-08-06
Description: 翻译 byte_tracker.py 的 STrack 与 BYTETracker 声明，补齐 xywh2ltwh 批量重载并保留 C++ 安全接口
    1.5: 新增 debugCallback 钩子（默认关闭，不改变行为），供应用层排查匹配代价明细。
    1.6: maxTimeLost 按 track_buffer 计 tracker 帧（max(1,buffer)），构造传入 skipN；
         debug 全量增强（vf 注入、状态事件、未匹配 cost、ID_BREAK/SWAP、lifecycle CSV、stats）。
*/

#ifndef TRACKER_CPP_BYTE_TRACKER_H
#define TRACKER_CPP_BYTE_TRACKER_H

#include "baseTrack.h"
#include "kalmanFilter.h"
#include "trackerConfig.h"

#include <Eigen/Dense>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trackerCpp {

// 单帧检测结果封装：支持普通包围框(xywh)和旋转包围框(xywhr)
struct DetectionResults {
    Eigen::MatrixXd xywh;    // 普通检测框 [N, 4] (x, y, w, h)
    Eigen::MatrixXd xywhr;   // 旋转检测框 [N, 5] (x, y, w, h, angle)
    Eigen::VectorXd conf;    // 检测置信度 [N]
    Eigen::VectorXd cls;     // 类别 ID [N]
    bool hasXywhr = false;   // 是否使用旋转框
};

// 将中心坐标格式 (x,y,w,h) 转换为左上角格式 (left,top,w,h)
Eigen::VectorXd xywh2ltwh(const Eigen::VectorXd& x);
// 批量版本：对矩阵的每一行执行 xywh -> ltwh 转换
Eigen::MatrixXd xywh2ltwh(const Eigen::MatrixXd& x);

class STrack;
using TrackPtr = std::shared_ptr<STrack>;
// GMC 全局运动补偿回调：输入图像和检测框，输出 2x3 仿射变换矩阵
using GmcFunction = std::function<Eigen::Matrix<double, 2, 3>(const Eigen::MatrixXd& img, const Eigen::MatrixXd& dets)>;

class STrack : public BaseTrack {
public:
    STrack();
    STrack(const Eigen::VectorXd& xywh, double score, double cls);

    void predict() override;                                           // 使用卡尔曼滤波预测下一帧状态
    static void multiPredict(std::vector<TrackPtr>& stracks);          // 批量预测多条轨迹
    static void multiGmc(std::vector<TrackPtr>& stracks, const Eigen::Matrix<double, 2, 3>& h);  // 批量全局运动补偿

    void activate(KalmanFilterXYAH* kalmanFilter, int frameId);        // 激活新轨迹（分配 ID 并初始化卡尔曼状态）
    void reActivate(const STrack& newTrack, int frameId, bool newId = false);  // 重新激活丢失的轨迹
    void update(const STrack& newTrack, int frameId);                  // 用新的检测结果更新当前轨迹

    Eigen::VectorXd convertCoords(const Eigen::VectorXd& tlwh) const;  // 坐标转换（tlwh -> xyah）
    Eigen::VectorXd tlwh() const;                                      // 获取当前 tlwh 估计值
    Eigen::VectorXd xyxy() const;                                      // 获取左上角+右下角格式 (x1,y1,x2,y2)
    Eigen::VectorXd detXyxy() const;                                   // 获取原始检测的 xyxy
    Eigen::VectorXd detXywh() const;                                   // 获取原始检测的 xywh
    Eigen::VectorXd detXywha() const;                                  // 获取原始检测的旋转框 xywha
    static Eigen::VectorXd tlwhToXyah(const Eigen::VectorXd& tlwh);    // 静态工具：tlwh 转 xyah
    Eigen::VectorXd xywh() const;                                      // 获取当前 xywh 估计值
    Eigen::VectorXd xywha() const;                                     // 获取当前旋转框 xywha
    std::vector<double> result() const;                                // 输出轨迹结果向量
    std::string repr() const;                                          // 轨迹字符串表示

    static KalmanFilterXYAH sharedKalman;    // 所有 STrack 共享的卡尔曼滤波器实例

    Eigen::VectorXd tlwhValue;               // 当前 tlwh 估计值（来自卡尔曼或检测）
    Eigen::VectorXd detTlwh;                 // 最近一次检测的原始 tlwh
    KalmanFilterXYAH* kalmanFilter;          // 指向本轨迹使用的卡尔曼滤波器
    Eigen::VectorXd mean;                    // 卡尔曼状态均值 [8] (x,y,a,h,vx,vy,va,vh)
    Eigen::MatrixXd covariance;              // 卡尔曼状态协方差 [8x8]
    int trackletLen;                         // 轨迹已连续追踪的帧数
    double cls;                              // 类别 ID
    double idx;                              // 检测来源索引
    std::optional<double> angle;             // 旋转框角度（仅 OBB 有效）
};

class BYTETracker {
public:
    // frameRate：有效处理帧率（通常 video_fps/N，仅日志）；skipN：启动跳帧参数
    explicit BYTETracker(const TrackerConfig& args, int frameRate = 30, int skipN = 1);
    BYTETracker(const BYTETracker&) = delete;
    BYTETracker& operator=(const BYTETracker&) = delete;
    BYTETracker(BYTETracker&&) = delete;
    BYTETracker& operator=(BYTETracker&&) = delete;

    // 核心更新函数：输入单帧检测结果，返回当前所有激活轨迹的矩阵
    Eigen::MatrixXf update(
        const DetectionResults& results,
        const Eigen::MatrixXd* img = nullptr,
        const Eigen::MatrixXd* feats = nullptr
    );

    KalmanFilterXYAH getKalmanFilter() const;                          // 获取默认卡尔曼滤波器实例
    std::vector<TrackPtr> initTrack(
        const Eigen::MatrixXd& dets,
        const Eigen::VectorXd& scores,
        const Eigen::VectorXd& cls,
        const Eigen::MatrixXd* img = nullptr
    ) const;                                                           // 将检测结果初始化为 STrack 列表
    Eigen::MatrixXd getDists(const std::vector<TrackPtr>& tracks, const std::vector<TrackPtr>& detections) const;  // 计算轨迹与检测之间的距离矩阵
    void multiPredict(std::vector<TrackPtr>& tracks) const;             // 批量预测
    static void resetId();                                             // 重置全局轨迹 ID 计数器
    void reset();                                                      // 重置追踪器全部状态

    // 调试钩子：设置后 update() 会回调输出高/低分划分、三轮匹配代价与耗时、轨迹状态统计；
    // 默认空（不输出、不改变任何行为）。仅供排查追踪问题使用。
    using DebugCallback = std::function<void(const std::string&)>;
    void setDebugCallback(DebugCallback cb);

    // 应用层在 update 前注入当前视频帧号（debug 日志 vf=）；-1 表示未知
    void setCurrentVideoFrameId(int64_t videoFrameId) { currentVideoFrameId = videoFrameId; }
    int64_t currentVideoFrameIdValue() const { return currentVideoFrameId; }

    int skipNValue() const { return skipN; }
    int maxTimeLostValue() const { return maxTimeLost; }
    int trackerFrameRateValue() const { return trackerFrameRate; }

    // debug 结束：汇总 stats；可选写出 lifecycle CSV
    void logFinalStats() const;
    bool dumpLifecycleCsv(const std::string& path) const;

    static std::vector<TrackPtr> jointStracks(const std::vector<TrackPtr>& tlista, const std::vector<TrackPtr>& tlistb);  // 合并两个轨迹列表（去重）
    static std::vector<TrackPtr> subStracks(const std::vector<TrackPtr>& tlista, const std::vector<TrackPtr>& tlistb);   // 轨迹列表差集
    static std::pair<std::vector<TrackPtr>, std::vector<TrackPtr>> removeDuplicateStracks(
        const std::vector<TrackPtr>& stracksa,
        const std::vector<TrackPtr>& stracksb
    );                                                                 // 移除 tracked 与 lost 中的重复轨迹

    std::vector<TrackPtr> trackedStracks;   // 当前正在追踪的轨迹列表
    std::vector<TrackPtr> lostStracks;      // 暂时丢失的轨迹列表
    std::vector<TrackPtr> removedStracks;   // 已移除的轨迹列表（保留用于查询）
    int frameId;                            // 当前处理帧号
    TrackerConfig args;                     // 追踪器配置参数
    int maxTimeLost;                        // 轨迹允许的最大丢失帧数（tracker 帧）
    int skipN;                              // 启动跳帧参数 N
    int trackerFrameRate;                   // 构造时 frameRate（日志用）
    KalmanFilterXYAH kalmanFilter;          // 本追踪器实例的卡尔曼滤波器
    GmcFunction gmc;                        // 全局运动补偿回调函数
    DebugCallback debugCallback;            // 调试输出回调（默认空）

private:
    struct RecentRemoved {
        int trackId = 0;
        int tf = 0;
        int64_t vf = -1;
        double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        double score = 0;
        std::string reason;
    };

    struct LifecycleRec {
        int trackId = 0;
        int birthTf = 0;
        int deathTf = -1;
        int64_t birthVf = -1;
        int64_t deathVf = -1;
        std::string deathReason;
        int numUpdates = 0;
        double sumCx = 0, sumCy = 0, sumW = 0, sumH = 0;
        double maxScore = 0;
        double minScore = 1e9;
        bool dead = false;
    };

    struct DebugStats {
        int totalNew = 0;
        int totalRemove = 0;
        int totalRefind = 0;
        int totalLost = 0;
        int idBreak = 0;
        int idSwap = 0;
        int lostNoDet = 0;
        int lostNoMatch = 0;
        int removeTimeout = 0;
        int removeUnconfirmed = 0;
        int removeDup = 0;
        int removeOther = 0;
    };

    int64_t currentVideoFrameId = -1;
    bool initLogged = false;
    std::deque<RecentRemoved> recentRemoved;
    std::unordered_map<int, LifecycleRec> lifecycle;
    DebugStats stats{};

    void emitDebug(const std::string& msg) const;
    void logInitOnce();
    void noteBirth(const TrackPtr& track);
    void noteUpdate(const TrackPtr& track);
    void noteDeath(const TrackPtr& track, const std::string& reason);
    void pushRecentRemoved(const TrackPtr& track, const std::string& reason);
    void checkIdBreak(const TrackPtr& newTrack);
};

BYTETracker createByteTracker();

} // namespace trackerCpp

#endif // TRACKER_CPP_BYTE_TRACKER_H
