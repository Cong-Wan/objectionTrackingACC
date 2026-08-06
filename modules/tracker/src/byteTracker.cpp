/*
Author: wilbur
Version: 1.6
Date: 2026-08-06
Description: 翻译 byte_tracker.py 的 STrack 与 BYTETracker 实现，补齐 xywh2ltwh 批量重载、消除重载歧义并增加 GMC catch-all 容错
    1.5: update() 增加 debugCallback 插桩（高/低分划分、三轮匹配代价矩阵/对数/匈牙利耗时、轨迹状态统计），
    默认关闭不影响行为。
    1.6: maxTimeLost=max(1,track_buffer)（按 tracker 帧，构造传 skipN）；debug P0~P2：vf 注入、
    状态事件、pool/removed 明细、未匹配 top1 cost、cost 矩阵、ID_BREAK/SWAP、lifecycle CSV、stats。
*/

#include "byteTracker.h"

#include "matching.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace trackerCpp {
namespace {

constexpr int kRecentRemovedCap = 64;
constexpr int kIdBreakTfWindow = 60;
constexpr double kIdBreakIouThresh = 0.5;
constexpr int kCostMatrixMaxCells = 16;

std::string fmtBox4(const Eigen::VectorXd& xyxy) {
    char buf[96];
    if (xyxy.size() < 4) {
        return "(?)";
    }
    std::snprintf(buf, sizeof(buf), "(%.1f,%.1f,%.1f,%.1f)", xyxy(0), xyxy(1), xyxy(2), xyxy(3));
    return buf;
}

double boxIouXyxy(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    if (a.size() < 4 || b.size() < 4) {
        return 0.0;
    }
    const double xx1 = std::max(a(0), b(0));
    const double yy1 = std::max(a(1), b(1));
    const double xx2 = std::min(a(2), b(2));
    const double yy2 = std::min(a(3), b(3));
    const double w = std::max(0.0, xx2 - xx1);
    const double h = std::max(0.0, yy2 - yy1);
    const double inter = w * h;
    const double areaA = std::max(0.0, a(2) - a(0)) * std::max(0.0, a(3) - a(1));
    const double areaB = std::max(0.0, b(2) - b(0)) * std::max(0.0, b(3) - b(1));
    const double uni = areaA + areaB - inter;
    return uni > 1e-9 ? inter / uni : 0.0;
}

double centerDistXyxy(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    if (a.size() < 4 || b.size() < 4) {
        return 1e9;
    }
    const double acx = 0.5 * (a(0) + a(2));
    const double acy = 0.5 * (a(1) + a(3));
    const double bcx = 0.5 * (b(0) + b(2));
    const double bcy = 0.5 * (b(1) + b(3));
    const double dx = acx - bcx;
    const double dy = acy - bcy;
    return std::sqrt(dx * dx + dy * dy);
}

// 对 unmatched 侧找 top-1 cost（最小）
void findTop1(const Eigen::MatrixXd& cost, int row, int& bestCol, double& bestCost) {
    bestCol = -1;
    bestCost = 1e9;
    if (row < 0 || row >= cost.rows() || cost.cols() == 0) {
        return;
    }
    for (int c = 0; c < cost.cols(); ++c) {
        if (cost(row, c) < bestCost) {
            bestCost = cost(row, c);
            bestCol = c;
        }
    }
}

void findTop1Col(const Eigen::MatrixXd& cost, int col, int& bestRow, double& bestCost) {
    bestRow = -1;
    bestCost = 1e9;
    if (col < 0 || col >= cost.cols() || cost.rows() == 0) {
        return;
    }
    for (int r = 0; r < cost.rows(); ++r) {
        if (cost(r, col) < bestCost) {
            bestCost = cost(r, col);
            bestRow = r;
        }
    }
}

std::string fmtCostMatrix(const Eigen::MatrixXd& cost) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "[";
    for (int r = 0; r < cost.rows(); ++r) {
        if (r) os << ",";
        os << "[";
        for (int c = 0; c < cost.cols(); ++c) {
            if (c) os << ",";
            os << cost(r, c);
        }
        os << "]";
    }
    os << "]";
    return os.str();
}

// 从 DetectionResults 中提取统一格式的包围框矩阵
// 若 hasXywhr=true 则返回 xywhr（5列），否则返回 xywh（4列）
Eigen::MatrixXd normalizedBoxes(const DetectionResults& results) {
    const Eigen::MatrixXd& boxes = results.hasXywhr ? results.xywhr : results.xywh;
    const int expectedCols = results.hasXywhr ? 5 : 4;

    if (boxes.rows() == 0 && boxes.cols() == 0) {
        return Eigen::MatrixXd::Zero(0, expectedCols);
    }
    if (boxes.cols() != expectedCols) {
        throw std::invalid_argument(results.hasXywhr ? "xywhr must have 5 columns" : "xywh must have 4 columns");
    }
    return boxes;
}

// 校验检测结果的一致性：包围框行数必须与 conf、cls 向量长度相等
void validateDetectionResults(const DetectionResults& results, const Eigen::MatrixXd& boxes) {
    if (results.conf.size() != boxes.rows()) {
        throw std::invalid_argument("conf size must match detection rows");
    }
    if (results.cls.size() != boxes.rows()) {
        throw std::invalid_argument("cls size must match detection rows");
    }
}

// 在包围框矩阵末尾追加一列原始索引，用于后续追踪回溯
Eigen::MatrixXd appendIndexColumn(const Eigen::MatrixXd& boxes) {
    Eigen::MatrixXd result(boxes.rows(), boxes.cols() + 1);
    if (boxes.rows() == 0) {
        return result;
    }
    result.leftCols(boxes.cols()) = boxes;
    for (int i = 0; i < boxes.rows(); ++i) {
        result(i, boxes.cols()) = static_cast<double>(i);
    }
    return result;
}

// 按索引列表从矩阵中抽取指定行
Eigen::MatrixXd selectRows(const Eigen::MatrixXd& matrix, const std::vector<int>& rows) {
    Eigen::MatrixXd result(static_cast<int>(rows.size()), matrix.cols());
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        result.row(i) = matrix.row(rows[static_cast<size_t>(i)]);
    }
    return result;
}

// 按索引列表从向量中抽取指定元素
Eigen::VectorXd selectValues(const Eigen::VectorXd& values, const std::vector<int>& rows) {
    Eigen::VectorXd result(static_cast<int>(rows.size()));
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        result(i) = values(rows[static_cast<size_t>(i)]);
    }
    return result;
}

// 根据轨迹是否包含旋转角度，推断输出结果矩阵的列数（8 或 9 列）
int trackResultColumns(const std::vector<TrackPtr>& tracks, int fallbackCols) {
    for (const TrackPtr& track : tracks) {
        if (track && track->isActivated) {
            return track->angle.has_value() ? 9 : 8;
        }
    }
    return fallbackCols;
}

// 将轨迹列表转换为包围框矩阵，用于计算 IoU 距离
// 普通轨迹输出 xyxy (4列)，旋转轨迹输出 xywha (5列)
Eigen::MatrixXd tracksToBoxes(const std::vector<TrackPtr>& tracks) {
    if (tracks.empty()) {
        return Eigen::MatrixXd::Zero(0, 4);
    }

    bool anyWithAngle = std::any_of(tracks.begin(), tracks.end(), [](const TrackPtr& track) {
        return track->angle.has_value();
    });
    bool allWithAngle = std::all_of(tracks.begin(), tracks.end(), [](const TrackPtr& track) {
        return track->angle.has_value();
    });
    if (anyWithAngle != allWithAngle) {
        throw std::invalid_argument("mixed xyxy and xywha tracks are not supported");
    }

    const int cols = allWithAngle ? 5 : 4;
    Eigen::MatrixXd boxes(static_cast<int>(tracks.size()), cols);
    for (int i = 0; i < static_cast<int>(tracks.size()); ++i) {
        Eigen::VectorXd box = allWithAngle ? tracks[static_cast<size_t>(i)]->xywha() : tracks[static_cast<size_t>(i)]->xyxy();
        boxes.row(i) = box.transpose();
    }
    return boxes;
}

// 按索引列表从轨迹列表中抽取子集
std::vector<TrackPtr> selectTracks(const std::vector<TrackPtr>& tracks, const std::vector<int>& indices) {
    std::vector<TrackPtr> result;
    result.reserve(indices.size());
    for (int index : indices) {
        result.push_back(tracks[static_cast<size_t>(index)]);
    }
    return result;
}

// 将激活状态的轨迹列表转为输出矩阵 [N, 8] 或 [N, 9]
// 列格式：x,y,x2,y2,trackId,score,cls,idx （旋转框额外包含 angle）
Eigen::MatrixXf resultsToMatrix(const std::vector<TrackPtr>& tracks, int fallbackCols) {
    std::vector<std::vector<double>> rows;
    const int outputCols = trackResultColumns(tracks, fallbackCols);
    for (const TrackPtr& track : tracks) {
        if (!track->isActivated) {
            continue;
        }
        rows.push_back(track->result());
    }

    Eigen::MatrixXf result = Eigen::MatrixXf::Constant(
        static_cast<int>(rows.size()),
        outputCols,
        std::numeric_limits<float>::quiet_NaN()
    );
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        for (int j = 0; j < static_cast<int>(rows[static_cast<size_t>(i)].size()) && j < outputCols; ++j) {
            result(i, j) = static_cast<float>(rows[static_cast<size_t>(i)][static_cast<size_t>(j)]);
        }
    }
    return result;
}

// 校验轨迹是否已完成卡尔曼初始化，防止在 activate 前调用 predict/update
void ensureTrackReady(const STrack& track, const std::string& operation) {
    if (track.kalmanFilter == nullptr || track.mean.size() != 8 || track.covariance.rows() != 8 || track.covariance.cols() != 8) {
        throw std::logic_error(operation + " called before track activation");
    }
}

} // namespace

KalmanFilterXYAH STrack::sharedKalman;  // 所有 STrack 共享的静态卡尔曼滤波器

// 单元素版本：将 (x,y,w,h) 转换为 (left,top,w,h)
Eigen::VectorXd xywh2ltwh(const Eigen::VectorXd& x) {
    if (x.size() < 4) {
        throw std::invalid_argument("xywh2ltwh expects at least 4 values");
    }
    Eigen::VectorXd y = x;
    y(0) = x(0) - x(2) / 2.0;
    y(1) = x(1) - x(3) / 2.0;
    return y;
}

// 批量版本：对矩阵每一行执行 xywh -> ltwh 转换
Eigen::MatrixXd xywh2ltwh(const Eigen::MatrixXd& x) {
    if (x.cols() < 4) {
        throw std::invalid_argument("xywh2ltwh expects at least 4 columns");
    }
    Eigen::MatrixXd y = x;
    y.col(0) = x.col(0) - x.col(2) / 2.0;
    y.col(1) = x.col(1) - x.col(3) / 2.0;
    return y;
}

// 默认构造：用于占位或预分配
STrack::STrack()
    : kalmanFilter(nullptr),
      trackletLen(0),
      cls(0.0),
      idx(0.0) {}

// 用检测结果构造轨迹：解析 xywh（5列）或 xywha（6列）输入，计算 ltwh 并记录类别与置信度
STrack::STrack(const Eigen::VectorXd& xywhValue, double scoreValue, double clsValue)
    : kalmanFilter(nullptr),
      trackletLen(0),
      cls(clsValue),
      idx(0.0) {
    if (xywhValue.size() != 5 && xywhValue.size() != 6) {
        throw std::invalid_argument("expected 5 or 6 values for STrack input");
    }

    tlwhValue = xywh2ltwh(Eigen::VectorXd(xywhValue.head(4)));
    detTlwh = tlwhValue;
    mean.resize(0);
    covariance.resize(0, 0);
    isActivated = false;
    score = scoreValue;
    idx = xywhValue(xywhValue.size() - 1);
    angle = xywhValue.size() == 6 ? std::optional<double>(xywhValue(4)) : std::nullopt;
}

// 卡尔曼预测：若轨迹未处于 Tracked 状态，则将纵向速度置零以抑制漂移
void STrack::predict() {
    ensureTrackReady(*this, "STrack::predict");
    Eigen::VectorXd meanState = mean;
    if (state != TrackState::Tracked) {
        meanState(7) = 0.0;
    }
    auto predicted = kalmanFilter->predict(meanState, covariance);
    mean = predicted.first;
    covariance = predicted.second;
}

// 批量卡尔曼预测：将多条轨迹的状态堆叠为矩阵，调用 sharedKalman.multiPredict 一次性计算
void STrack::multiPredict(std::vector<TrackPtr>& stracks) {
    if (stracks.empty()) {
        return;
    }

    Eigen::MatrixXd multiMean(static_cast<int>(stracks.size()), 8);
    std::vector<Eigen::MatrixXd> multiCovariance;
    multiCovariance.reserve(stracks.size());

    for (int i = 0; i < static_cast<int>(stracks.size()); ++i) {
        ensureTrackReady(*stracks[static_cast<size_t>(i)], "STrack::multiPredict");
        Eigen::VectorXd meanCopy = stracks[static_cast<size_t>(i)]->mean;
        if (stracks[static_cast<size_t>(i)]->state != TrackState::Tracked) {
            meanCopy(7) = 0.0;
        }
        multiMean.row(i) = meanCopy.transpose();
        multiCovariance.push_back(stracks[static_cast<size_t>(i)]->covariance);
    }

    auto predicted = sharedKalman.multiPredict(multiMean, multiCovariance);
    for (int i = 0; i < static_cast<int>(stracks.size()); ++i) {
        stracks[static_cast<size_t>(i)]->mean = predicted.first.row(i).transpose();
        stracks[static_cast<size_t>(i)]->covariance = predicted.second[static_cast<size_t>(i)];
    }
}

// 批量全局运动补偿：将 2x3 仿射变换扩展为 8x8 状态空间变换，作用于 mean 和 covariance
void STrack::multiGmc(std::vector<TrackPtr>& stracks, const Eigen::Matrix<double, 2, 3>& h) {
    if (stracks.empty()) {
        return;
    }

    Eigen::Matrix2d r = h.block<2, 2>(0, 0);
    Eigen::Vector2d t = h.block<2, 1>(0, 2);
    Eigen::Matrix<double, 8, 8> r8x8 = Eigen::Matrix<double, 8, 8>::Zero();
    for (int i = 0; i < 4; ++i) {
        r8x8.block<2, 2>(2 * i, 2 * i) = r;
    }

    for (const TrackPtr& track : stracks) {
        ensureTrackReady(*track, "STrack::multiGmc");
        track->mean = r8x8 * track->mean;
        track->mean.head(2) += t;
        track->covariance = r8x8 * track->covariance * r8x8.transpose();
    }
}

// 激活新轨迹：分配唯一 ID，用当前 tlwh 初始化卡尔曼状态，首帧即激活
void STrack::activate(KalmanFilterXYAH* kalmanFilterValue, int frameIdValue) {
    if (kalmanFilterValue == nullptr) {
        throw std::invalid_argument("kalmanFilter must not be null");
    }
    kalmanFilter = kalmanFilterValue;
    trackId = nextId();
    auto initiated = kalmanFilter->initiate(convertCoords(tlwhValue));
    mean = initiated.first;
    covariance = initiated.second;

    trackletLen = 0;
    detTlwh = tlwhValue;
    state = TrackState::Tracked;
    if (frameIdValue == 1) {
        isActivated = true;
    }
    frameId = frameIdValue;
    startFrame = frameIdValue;
}

// 重新激活丢失轨迹：用新的检测结果更新卡尔曼状态，可选是否分配新 ID
void STrack::reActivate(const STrack& newTrack, int frameIdValue, bool newId) {
    ensureTrackReady(*this, "STrack::reActivate");
    Eigen::VectorXd newTlwh = newTrack.tlwh();
    auto updated = kalmanFilter->update(mean, covariance, convertCoords(newTlwh));
    mean = updated.first;
    covariance = updated.second;
    detTlwh = newTlwh;
    trackletLen = 0;
    state = TrackState::Tracked;
    isActivated = true;
    frameId = frameIdValue;
    if (newId) {
        trackId = nextId();
    }
    score = newTrack.score;
    cls = newTrack.cls;
    angle = newTrack.angle;
    idx = newTrack.idx;
}

// 正常更新：用匹配的检测结果更新卡尔曼状态，增加 trackletLen，重置状态为 Tracked
void STrack::update(const STrack& newTrack, int frameIdValue) {
    ensureTrackReady(*this, "STrack::update");
    frameId = frameIdValue;
    trackletLen += 1;

    Eigen::VectorXd newTlwh = newTrack.tlwh();
    auto updated = kalmanFilter->update(mean, covariance, convertCoords(newTlwh));
    mean = updated.first;
    covariance = updated.second;
    detTlwh = newTlwh;
    state = TrackState::Tracked;
    isActivated = true;

    score = newTrack.score;
    cls = newTrack.cls;
    angle = newTrack.angle;
    idx = newTrack.idx;
}

// 坐标转换接口：当前实现即将 tlwh 转为 xyah（供卡尔曼观测）
Eigen::VectorXd STrack::convertCoords(const Eigen::VectorXd& tlwhInput) const {
    return tlwhToXyah(tlwhInput);
}

// 从卡尔曼状态均值还原 tlwh：mean[2]*mean[3] 得到宽度，head(2) 为中心坐标
Eigen::VectorXd STrack::tlwh() const {
    if (mean.size() == 0) {
        return tlwhValue;
    }
    Eigen::VectorXd ret = mean.head(4);
    ret(2) *= ret(3);
    ret.head(2) -= ret.tail(2) / 2.0;
    return ret;
}

// 将 tlwh 转为 xyxy（左上角+右下角）格式
Eigen::VectorXd STrack::xyxy() const {
    Eigen::VectorXd ret = tlwh();
    ret.tail(2) += ret.head(2);
    return ret;
}

// 最近一次检测结果的 xyxy 格式
Eigen::VectorXd STrack::detXyxy() const {
    Eigen::VectorXd ret = detTlwh;
    ret.tail(2) += ret.head(2);
    return ret;
}

// 最近一次检测结果的 xywh 格式（中心坐标）
Eigen::VectorXd STrack::detXywh() const {
    Eigen::VectorXd ret = detTlwh;
    ret.head(2) += ret.tail(2) / 2.0;
    return ret;
}

// 最近一次检测结果的旋转框 xywha 格式；若无角度则退化为 xywh
Eigen::VectorXd STrack::detXywha() const {
    if (!angle.has_value()) {
        return detXywh();
    }
    Eigen::VectorXd ret(5);
    ret.head(4) = detXywh();
    ret(4) = angle.value();
    return ret;
}

// 静态工具函数：tlwh -> xyah（x_center, y_center, aspect_ratio=w/h, height）
Eigen::VectorXd STrack::tlwhToXyah(const Eigen::VectorXd& tlwhInput) {
    if (tlwhInput.size() != 4) {
        throw std::invalid_argument("tlwhToXyah expects 4 values");
    }
    Eigen::VectorXd ret = tlwhInput;
    ret.head(2) += ret.tail(2) / 2.0;
    ret(2) /= ret(3);
    return ret;
}

// 将 tlwh 转为 xywh（中心坐标）格式
Eigen::VectorXd STrack::xywh() const {
    Eigen::VectorXd ret = tlwh();
    ret.head(2) += ret.tail(2) / 2.0;
    return ret;
}

// 旋转框 xywha 格式；若无角度则退化为 xywh
Eigen::VectorXd STrack::xywha() const {
    if (!angle.has_value()) {
        return xywh();
    }
    Eigen::VectorXd ret(5);
    ret.head(4) = xywh();
    ret(4) = angle.value();
    return ret;
}

// 生成当前轨迹的输出向量：[box_coords(4 or 5), trackId, score, cls, idx]
std::vector<double> STrack::result() const {
    Eigen::VectorXd coords = angle.has_value() ? detXywha() : detXyxy();
    std::vector<double> values;
    values.reserve(static_cast<size_t>(coords.size()) + 4);
    for (int i = 0; i < coords.size(); ++i) {
        values.push_back(coords(i));
    }
    values.push_back(static_cast<double>(trackId));
    values.push_back(score);
    values.push_back(cls);
    values.push_back(idx);
    return values;
}

// 轨迹的字符串表示：OT_trackId_(startFrame-endFrame)
std::string STrack::repr() const {
    std::ostringstream stream;
    stream << "OT_" << trackId << "_(" << startFrame << "-" << endFrame() << ")";
    return stream.str();
}

// 构造追踪器：maxTimeLost 按 tracker 帧 = max(1, track_buffer)；skipN 仅用于时间换算日志
BYTETracker::BYTETracker(const TrackerConfig& argsValue, int frameRate, int skipNValue)
    : frameId(0),
      args(argsValue),
      maxTimeLost(std::max(1, argsValue.trackBuffer)),
      skipN(std::max(1, skipNValue)),
      trackerFrameRate(std::max(1, frameRate)),
      kalmanFilter(getKalmanFilter()),
      currentVideoFrameId(-1),
      initLogged(false) {
    resetId();
}

void BYTETracker::setDebugCallback(DebugCallback cb) {
    debugCallback = std::move(cb);
    initLogged = false;
    if (debugCallback) {
        logInitOnce();
    }
}

void BYTETracker::emitDebug(const std::string& msg) const {
    if (debugCallback) {
        debugCallback(msg);
    }
}

void BYTETracker::logInitOnce() {
    if (initLogged || !debugCallback) {
        return;
    }
    initLogged = true;
    // trackerFrameRate 通常为 video_fps/N，真实秒数 = maxTimeLost / trackerFrameRate
    const double maxTimeLostSeconds =
        static_cast<double>(maxTimeLost) / static_cast<double>(std::max(1, trackerFrameRate));
    const int maxTimeLostVideoFrames = maxTimeLost * skipN;
    std::ostringstream os;
    os << std::fixed;
    os << "tracker init: N=" << skipN
       << " tracker_frameRate=" << trackerFrameRate
       << " track_buffer=" << args.trackBuffer
       << " maxTimeLost=" << maxTimeLost
       << " (video_frames=" << maxTimeLostVideoFrames
       << ", seconds=" << std::setprecision(3) << maxTimeLostSeconds << ")"
       << " high=" << args.trackHighThresh
       << " low=" << args.trackLowThresh
       << " new=" << args.newTrackThresh
       << " match=" << args.matchThresh
       << " fuse_score=" << (args.fuseScore ? 1 : 0)
       << " mot20=" << (args.mot20 ? 1 : 0)
       << " aspect_ratio_thresh=" << args.aspectRatioThresh
       << " min_box_area=" << args.minBoxArea;
    emitDebug(os.str());
}

void BYTETracker::noteBirth(const TrackPtr& track) {
    if (!track) return;
    LifecycleRec rec;
    rec.trackId = track->trackId;
    rec.birthTf = frameId;
    rec.birthVf = currentVideoFrameId;
    rec.numUpdates = 0;
    rec.maxScore = track->score;
    rec.minScore = track->score;
    Eigen::VectorXd box = track->xyxy();
    if (box.size() >= 4) {
        rec.sumCx = 0.5 * (box(0) + box(2));
        rec.sumCy = 0.5 * (box(1) + box(3));
        rec.sumW = box(2) - box(0);
        rec.sumH = box(3) - box(1);
        rec.numUpdates = 1;
    }
    lifecycle[track->trackId] = rec;
    ++stats.totalNew;
}

void BYTETracker::noteUpdate(const TrackPtr& track) {
    if (!track) return;
    auto it = lifecycle.find(track->trackId);
    if (it == lifecycle.end()) {
        noteBirth(track);
        it = lifecycle.find(track->trackId);
    }
    LifecycleRec& rec = it->second;
    Eigen::VectorXd box = track->xyxy();
    if (box.size() >= 4) {
        rec.sumCx += 0.5 * (box(0) + box(2));
        rec.sumCy += 0.5 * (box(1) + box(3));
        rec.sumW += box(2) - box(0);
        rec.sumH += box(3) - box(1);
        ++rec.numUpdates;
    }
    rec.maxScore = std::max(rec.maxScore, track->score);
    rec.minScore = std::min(rec.minScore, track->score);
}

void BYTETracker::noteDeath(const TrackPtr& track, const std::string& reason) {
    if (!track) return;
    auto it = lifecycle.find(track->trackId);
    if (it == lifecycle.end()) {
        noteBirth(track);
        it = lifecycle.find(track->trackId);
    }
    if (it->second.dead) {
        return;  // 避免同一 id 重复计入 remove 统计
    }
    it->second.deathTf = frameId;
    it->second.deathVf = currentVideoFrameId;
    it->second.deathReason = reason;
    it->second.dead = true;
    ++stats.totalRemove;
    if (reason == "timeout") ++stats.removeTimeout;
    else if (reason == "unconfirmed_fail") ++stats.removeUnconfirmed;
    else if (reason == "dup") ++stats.removeDup;
    else ++stats.removeOther;
}

void BYTETracker::pushRecentRemoved(const TrackPtr& track, const std::string& reason) {
    if (!track) return;
    RecentRemoved rr;
    rr.trackId = track->trackId;
    rr.tf = frameId;
    rr.vf = currentVideoFrameId;
    rr.score = track->score;
    rr.reason = reason;
    Eigen::VectorXd box = track->xyxy();
    if (box.size() >= 4) {
        rr.x1 = box(0); rr.y1 = box(1); rr.x2 = box(2); rr.y2 = box(3);
    }
    recentRemoved.push_back(rr);
    while (static_cast<int>(recentRemoved.size()) > kRecentRemovedCap) {
        recentRemoved.pop_front();
    }
}

void BYTETracker::checkIdBreak(const TrackPtr& newTrack) {
    if (!newTrack || !debugCallback) return;
    Eigen::VectorXd nbox = newTrack->xyxy();
    if (nbox.size() < 4) return;
    double bestIou = 0.0;
    const RecentRemoved* best = nullptr;
    for (const RecentRemoved& rr : recentRemoved) {
        if (frameId - rr.tf > kIdBreakTfWindow) continue;
        Eigen::VectorXd obox(4);
        obox << rr.x1, rr.y1, rr.x2, rr.y2;
        const double iou = boxIouXyxy(nbox, obox);
        if (iou >= kIdBreakIouThresh && iou > bestIou) {
            bestIou = iou;
            best = &rr;
        }
    }
    if (!best) return;
    ++stats.idBreak;
    std::ostringstream os;
    os << std::fixed << std::setprecision(2);
    os << "tracker tf=" << frameId << " vf=" << currentVideoFrameId
       << " ID_BREAK prev_id=" << best->trackId
       << " new_id=" << newTrack->trackId
       << " iou=" << bestIou
       << " dt_tf=" << (frameId - best->tf)
       << " dt_vf=" << (currentVideoFrameId >= 0 && best->vf >= 0 ? (currentVideoFrameId - best->vf) : -1)
       << " prev_box=(" << best->x1 << "," << best->y1 << "," << best->x2 << "," << best->y2 << ")"
       << " new_box=" << fmtBox4(nbox)
       << " prev_reason=" << best->reason;
    emitDebug(os.str());
}

// BYTETracker 核心更新流程（匹配策略不变）：
// 1. 高/低分划分  2. Round1 tracked+lost×high  3. Round2 tracked×low
// 4. Round3 unconfirmed×unmatched_high  5. NEW  6. timeout remove
Eigen::MatrixXf BYTETracker::update(const DetectionResults& results, const Eigen::MatrixXd* img, const Eigen::MatrixXd* feats) {
    Eigen::MatrixXd rawBoxes = normalizedBoxes(results);
    validateDetectionResults(results, rawBoxes);
    const int fallbackResultCols = results.hasXywhr ? 9 : 8;

    frameId += 1;
    logInitOnce();

    const bool dbg = static_cast<bool>(debugCallback);
    auto emit = [this](const std::string& msg) { emitDebug(msg); };
    auto tfVf = [this]() {
        std::ostringstream os;
        os << "tracker tf=" << frameId << " vf=" << currentVideoFrameId;
        return os.str();
    };
    auto nowMs = []() { return std::chrono::steady_clock::now(); };
    auto elapsedMs = [](auto t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    // 本帧事件明细（仅 debug 用于 REMOVE reason 映射）
    std::map<int, std::string> removeReasons;
    std::map<int, std::string> lostReasons;

    std::vector<TrackPtr> activatedStracks;
    std::vector<TrackPtr> refindStracks;
    std::vector<TrackPtr> lostStracksLocal;
    std::vector<TrackPtr> removedStracksLocal;

    Eigen::VectorXd scores = results.conf;
    Eigen::MatrixXd bboxes = appendIndexColumn(rawBoxes);
    Eigen::VectorXd cls = results.cls;

    std::vector<int> remainInds;
    std::vector<int> secondInds;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores(i) >= args.trackHighThresh) {
            remainInds.push_back(i);
        }
        if (scores(i) > args.trackLowThresh && scores(i) < args.trackHighThresh) {
            secondInds.push_back(i);
        }
    }

    if (dbg) {
        emit(tfVf() + ": input dets=" + std::to_string(scores.size()) +
             " high(>=" + std::to_string(args.trackHighThresh) + ")=" + std::to_string(remainInds.size()) +
             " low(" + std::to_string(args.trackLowThresh) + "~" + std::to_string(args.trackHighThresh) + ")=" +
             std::to_string(secondInds.size()));
    }

    Eigen::MatrixXd detsSecond = selectRows(bboxes, secondInds);
    Eigen::MatrixXd dets = selectRows(bboxes, remainInds);
    Eigen::VectorXd scoresKeep = selectValues(scores, remainInds);
    Eigen::VectorXd scoresSecond = selectValues(scores, secondInds);
    Eigen::VectorXd clsKeep = selectValues(cls, remainInds);
    Eigen::VectorXd clsSecond = selectValues(cls, secondInds);

    std::vector<TrackPtr> detections = initTrack(dets, scoresKeep, clsKeep, feats == nullptr ? img : feats);

    std::vector<TrackPtr> unconfirmed;
    std::vector<TrackPtr> trackedStracksLocal;
    for (const TrackPtr& track : trackedStracks) {
        if (!track->isActivated) {
            unconfirmed.push_back(track);
        } else {
            trackedStracksLocal.push_back(track);
        }
    }

    // 保存 pred 前 box 用于 ID_SWAP（仅 debug）
    std::map<int, Eigen::VectorXd> prevBoxes;
    if (dbg) {
        for (const TrackPtr& t : trackedStracksLocal) {
            prevBoxes[t->trackId] = t->xyxy();
        }
    }

    std::vector<TrackPtr> strackPool = jointStracks(trackedStracksLocal, lostStracks);
    multiPredict(strackPool);
    if (dbg) {
        emit(tfVf() + ": pool confirmed=" + std::to_string(trackedStracksLocal.size()) +
             " + lost=" + std::to_string(lostStracks.size()) +
             " = " + std::to_string(strackPool.size()) +
             ", unconfirmed=" + std::to_string(unconfirmed.size()));
        std::ostringstream pd;
        pd << tfVf() << " pool_detail: tracked=[";
        for (size_t i = 0; i < trackedStracksLocal.size(); ++i) {
            if (i) pd << ",";
            pd << "id=" << trackedStracksLocal[i]->trackId;
        }
        pd << "] lost=[";
        for (size_t i = 0; i < lostStracks.size(); ++i) {
            if (i) pd << ",";
            const TrackPtr& t = lostStracks[i];
            // lost 也在 strackPool 中被 predict 过，xyxy 已是 pred
            pd << "id=" << t->trackId << " end_tf=" << t->endFrame()
               << " age=" << (frameId - t->endFrame())
               << " pred=" << fmtBox4(t->xyxy());
        }
        pd << "] unconfirmed=[";
        for (size_t i = 0; i < unconfirmed.size(); ++i) {
            if (i) pd << ",";
            pd << "id=" << unconfirmed[i]->trackId;
        }
        pd << "]";
        emit(pd.str());
    }

    if (gmc && img != nullptr) {
        Eigen::Matrix<double, 2, 3> warp;
        try {
            warp = gmc(*img, dets);
        } catch (const std::exception&) {
            warp << 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0;
        } catch (...) {
            warp << 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0;
        }
        STrack::multiGmc(strackPool, warp);
        STrack::multiGmc(unconfirmed, warp);
    }

    // Round1: tracked+lost × high
    auto tR1 = nowMs();
    Eigen::MatrixXd distsR1 = getDists(strackPool, detections);
    AssignmentResult assignment = linearAssignment(distsR1, args.matchThresh);
    const char* distTag = args.fuseScore ? "iouDist+score" : "iouDist only";
    if (dbg) {
        emit(tfVf() + " round1 (tracked+lost × high): cost " +
             std::to_string(strackPool.size()) + "x" + std::to_string(detections.size()) +
             " (" + distTag + ", thresh=" + std::to_string(args.matchThresh) + "), hungarian=" +
             std::to_string(elapsedMs(tR1)) + " ms, matches=" + std::to_string(assignment.matches.size()) +
             " unmatchedTracks=" + std::to_string(assignment.unmatchedA.size()) +
             " unmatchedDets=" + std::to_string(assignment.unmatchedB.size()));
        if (strackPool.size() * detections.size() > 0 &&
            static_cast<int>(strackPool.size() * detections.size()) <= kCostMatrixMaxCells &&
            (strackPool.size() > 1 || detections.size() > 1)) {
            std::ostringstream ord;
            ord << "  round1 track_order=[";
            for (size_t i = 0; i < strackPool.size(); ++i) {
                if (i) ord << ",";
                ord << strackPool[i]->trackId;
            }
            ord << "] det_order=[";
            for (size_t i = 0; i < detections.size(); ++i) {
                if (i) ord << ",";
                ord << static_cast<int>(detections[i]->idx);
            }
            ord << "] cost_matrix=" << fmtCostMatrix(distsR1);
            emit(ord.str());
        }
        for (const auto& m : assignment.matches) {
            const TrackPtr& tr = strackPool[static_cast<size_t>(m.first)];
            const TrackPtr& det = detections[static_cast<size_t>(m.second)];
            const double cost = distsR1(m.first, m.second);
            const double iou = std::max(0.0, 1.0 - cost); // approximate when fuse_score off
            std::ostringstream os;
            os << std::fixed << std::setprecision(6);
            os << "  round1 match: trackId=" << tr->trackId
               << " <- det#" << static_cast<int>(det->idx)
               << " score=" << det->score
               << " cost=" << cost
               << " iou~=" << iou
               << " box=" << fmtBox4(det->xyxy());
            emit(os.str());
        }
        for (int ti : assignment.unmatchedA) {
            const TrackPtr& tr = strackPool[static_cast<size_t>(ti)];
            int bc = -1; double bcost = 1e9;
            findTop1(distsR1, ti, bc, bcost);
            std::ostringstream os;
            os << std::fixed << std::setprecision(4);
            os << "  round1 unmatched track id=" << tr->trackId
               << " pred=" << fmtBox4(tr->xyxy());
            if (bc >= 0) {
                const TrackPtr& det = detections[static_cast<size_t>(bc)];
                os << " top1_det#" << static_cast<int>(det->idx)
                   << " score=" << det->score
                   << " cost=" << bcost
                   << " iou~=" << std::max(0.0, 1.0 - bcost)
                   << " center_dist=" << centerDistXyxy(tr->xyxy(), det->xyxy())
                   << " (thresh=" << args.matchThresh << ")";
            } else if (detections.empty()) {
                os << " top1=none (no high dets)";
            }
            emit(os.str());
        }
        for (int di : assignment.unmatchedB) {
            const TrackPtr& det = detections[static_cast<size_t>(di)];
            int br = -1; double bcost = 1e9;
            findTop1Col(distsR1, di, br, bcost);
            std::ostringstream os;
            os << std::fixed << std::setprecision(4);
            os << "  round1 unmatched high det#" << static_cast<int>(det->idx)
               << " score=" << det->score
               << " box=" << fmtBox4(det->xyxy());
            if (br >= 0) {
                os << " top1_track id=" << strackPool[static_cast<size_t>(br)]->trackId
                   << " cost=" << bcost;
            }
            emit(os.str());
        }
    }

    for (const auto& match : assignment.matches) {
        TrackPtr track = strackPool[static_cast<size_t>(match.first)];
        const TrackPtr& det = detections[static_cast<size_t>(match.second)];
        const double cost = distsR1(match.first, match.second);
        if (track->state == TrackState::Tracked) {
            const bool wasUnconf = !track->isActivated;
            track->update(*det, frameId);
            activatedStracks.push_back(track);
            noteUpdate(track);
            if (dbg) {
                if (wasUnconf) {
                    emit(tfVf() + " EVT CONFIRM id=" + std::to_string(track->trackId) +
                         " round=1 score=" + std::to_string(track->score) +
                         " box=" + fmtBox4(track->xyxy()));
                } else {
                    emit(tfVf() + " EVT UPDATE id=" + std::to_string(track->trackId) +
                         " round=1 det#" + std::to_string(static_cast<int>(det->idx)) +
                         " score=" + std::to_string(det->score) +
                         " cost=" + std::to_string(cost) +
                         " box=" + fmtBox4(track->xyxy()));
                }
            }
        } else {
            track->reActivate(*det, frameId, false);
            refindStracks.push_back(track);
            noteUpdate(track);
            ++stats.totalRefind;
            if (dbg) {
                const int lostAge = frameId - track->startFrame; // not ideal; use end before reactivate unavailable
                emit(tfVf() + " EVT REFIND id=" + std::to_string(track->trackId) +
                     " det#" + std::to_string(static_cast<int>(det->idx)) +
                     " score=" + std::to_string(det->score) +
                     " cost=" + std::to_string(cost) +
                     " box=" + fmtBox4(track->xyxy()));
                (void)lostAge;
            }
        }
    }

    // Round2: tracked_only × low
    std::vector<TrackPtr> detectionsSecond = initTrack(detsSecond, scoresSecond, clsSecond, feats == nullptr ? img : feats);
    std::vector<TrackPtr> rTrackedStracks;
    std::vector<TrackPtr> unmatchedLostFromR1;
    for (int index : assignment.unmatchedA) {
        if (strackPool[static_cast<size_t>(index)]->state == TrackState::Tracked) {
            rTrackedStracks.push_back(strackPool[static_cast<size_t>(index)]);
        } else {
            unmatchedLostFromR1.push_back(strackPool[static_cast<size_t>(index)]);
        }
    }

    auto tR2 = nowMs();
    Eigen::MatrixXd distsR2 = iouDistance(tracksToBoxes(rTrackedStracks), tracksToBoxes(detectionsSecond));
    AssignmentResult secondAssignment = linearAssignment(distsR2, 0.5);
    if (dbg) {
        emit(tfVf() + " round2 (tracked_only × low): cost " +
             std::to_string(rTrackedStracks.size()) + "x" + std::to_string(detectionsSecond.size()) +
             " (iouDist, thresh=0.5), hungarian=" + std::to_string(elapsedMs(tR2)) +
             " ms, matches=" + std::to_string(secondAssignment.matches.size()) +
             " unmatchedTracks=" + std::to_string(secondAssignment.unmatchedA.size()) +
             " unmatchedLowDets=" + std::to_string(secondAssignment.unmatchedB.size()));
        for (const TrackPtr& lt : unmatchedLostFromR1) {
            emit(tfVf() + " round2 skip lost id=" + std::to_string(lt->trackId) +
                 " (lost tracks excluded from low-det matching by design) pred=" + fmtBox4(lt->xyxy()));
        }
        if (rTrackedStracks.size() * detectionsSecond.size() > 0 &&
            static_cast<int>(rTrackedStracks.size() * detectionsSecond.size()) <= kCostMatrixMaxCells &&
            (rTrackedStracks.size() > 1 || detectionsSecond.size() > 1)) {
            emit("  round2 cost_matrix=" + fmtCostMatrix(distsR2));
        }
        for (const auto& m : secondAssignment.matches) {
            const TrackPtr& det = detectionsSecond[static_cast<size_t>(m.second)];
            emit("  round2 match: trackId=" + std::to_string(rTrackedStracks[static_cast<size_t>(m.first)]->trackId) +
                 " <- lowDet#" + std::to_string(static_cast<int>(det->idx)) +
                 " score=" + std::to_string(det->score) +
                 " cost=" + std::to_string(distsR2(m.first, m.second)));
        }
        for (int ti : secondAssignment.unmatchedA) {
            const TrackPtr& tr = rTrackedStracks[static_cast<size_t>(ti)];
            int bc = -1; double bcost = 1e9;
            findTop1(distsR2, ti, bc, bcost);
            std::ostringstream os;
            os << std::fixed << std::setprecision(4);
            os << "  round2 unmatched track id=" << tr->trackId
               << " pred=" << fmtBox4(tr->xyxy());
            if (bc >= 0) {
                const TrackPtr& det = detectionsSecond[static_cast<size_t>(bc)];
                os << " top1_lowDet#" << static_cast<int>(det->idx)
                   << " score=" << det->score
                   << " cost=" << bcost
                   << " iou=" << std::max(0.0, 1.0 - bcost)
                   << " center_dist=" << centerDistXyxy(tr->xyxy(), det->xyxy());
            } else if (detectionsSecond.empty()) {
                os << " top1=none (no low dets)";
            }
            emit(os.str());
        }
    }
    for (const auto& match : secondAssignment.matches) {
        TrackPtr track = rTrackedStracks[static_cast<size_t>(match.first)];
        const TrackPtr& det = detectionsSecond[static_cast<size_t>(match.second)];
        const double cost = distsR2(match.first, match.second);
        if (track->state == TrackState::Tracked) {
            track->update(*det, frameId);
            activatedStracks.push_back(track);
            noteUpdate(track);
            if (dbg) {
                emit(tfVf() + " EVT UPDATE id=" + std::to_string(track->trackId) +
                     " round=2 det#" + std::to_string(static_cast<int>(det->idx)) +
                     " score=" + std::to_string(det->score) +
                     " cost=" + std::to_string(cost) +
                     " box=" + fmtBox4(track->xyxy()));
            }
        } else {
            track->reActivate(*det, frameId, false);
            refindStracks.push_back(track);
            noteUpdate(track);
            ++stats.totalRefind;
            if (dbg) {
                emit(tfVf() + " EVT REFIND id=" + std::to_string(track->trackId) +
                     " round=2 det#" + std::to_string(static_cast<int>(det->idx)) +
                     " score=" + std::to_string(det->score));
            }
        }
    }

    for (int index : secondAssignment.unmatchedA) {
        TrackPtr track = rTrackedStracks[static_cast<size_t>(index)];
        if (track->state != TrackState::Lost) {
            track->markLost();
            lostStracksLocal.push_back(track);
            const std::string reason = (scores.size() == 0) ? "no_det" : "no_match";
            lostReasons[track->trackId] = reason;
            ++stats.totalLost;
            if (reason == "no_det") ++stats.lostNoDet;
            else ++stats.lostNoMatch;
            if (dbg) {
                emit(tfVf() + " EVT LOST id=" + std::to_string(track->trackId) +
                     " reason=" + reason +
                     " last_score=" + std::to_string(track->score) +
                     " pred=" + fmtBox4(track->xyxy()));
            }
        }
    }

    // Round3: unconfirmed × unmatched high
    std::vector<TrackPtr> unmatchedDetections = selectTracks(detections, assignment.unmatchedB);
    auto tR3 = nowMs();
    Eigen::MatrixXd distsR3 = getDists(unconfirmed, unmatchedDetections);
    AssignmentResult unconfirmedAssignment = linearAssignment(distsR3, 0.7);
    if (dbg) {
        emit(tfVf() + " round3 (unconfirmed × unmatched_high): cost " +
             std::to_string(unconfirmed.size()) + "x" + std::to_string(unmatchedDetections.size()) +
             " (" + distTag + ", thresh=0.7), hungarian=" + std::to_string(elapsedMs(tR3)) +
             " ms, matches=" + std::to_string(unconfirmedAssignment.matches.size()) +
             " unmatchedUnconfirmed(removed)=" + std::to_string(unconfirmedAssignment.unmatchedA.size()) +
             " unmatchedDets=" + std::to_string(unconfirmedAssignment.unmatchedB.size()));
        for (const auto& m : unconfirmedAssignment.matches) {
            const TrackPtr& det = unmatchedDetections[static_cast<size_t>(m.second)];
            emit("  round3 match: trackId=" + std::to_string(unconfirmed[static_cast<size_t>(m.first)]->trackId) +
                 " <- det#" + std::to_string(static_cast<int>(det->idx)) +
                 " score=" + std::to_string(det->score) +
                 " cost=" + std::to_string(distsR3(m.first, m.second)));
        }
        for (int ti : unconfirmedAssignment.unmatchedA) {
            const TrackPtr& tr = unconfirmed[static_cast<size_t>(ti)];
            int bc = -1; double bcost = 1e9;
            findTop1(distsR3, ti, bc, bcost);
            std::ostringstream os;
            os << std::fixed << std::setprecision(4);
            os << "  round3 unmatched unconfirmed id=" << tr->trackId
               << " box=" << fmtBox4(tr->xyxy());
            if (bc >= 0) {
                os << " top1_det#" << static_cast<int>(unmatchedDetections[static_cast<size_t>(bc)]->idx)
                   << " cost=" << bcost;
            }
            emit(os.str());
        }
    }
    for (const auto& match : unconfirmedAssignment.matches) {
        TrackPtr track = unconfirmed[static_cast<size_t>(match.first)];
        const TrackPtr& det = unmatchedDetections[static_cast<size_t>(match.second)];
        track->update(*det, frameId);
        activatedStracks.push_back(track);
        noteUpdate(track);
        if (dbg) {
            emit(tfVf() + " EVT CONFIRM id=" + std::to_string(track->trackId) +
                 " round=3 score=" + std::to_string(track->score) +
                 " box=" + fmtBox4(track->xyxy()));
        }
    }

    for (int index : unconfirmedAssignment.unmatchedA) {
        TrackPtr track = unconfirmed[static_cast<size_t>(index)];
        track->markRemoved();
        removedStracksLocal.push_back(track);
        removeReasons[track->trackId] = "unconfirmed_fail";
        noteDeath(track, "unconfirmed_fail");
        pushRecentRemoved(track, "unconfirmed_fail");
        if (dbg) {
            emit(tfVf() + " EVT REMOVE id=" + std::to_string(track->trackId) +
                 " reason=unconfirmed_fail last_score=" + std::to_string(track->score) +
                 " box=" + fmtBox4(track->xyxy()));
        }
    }

    for (int index : unconfirmedAssignment.unmatchedB) {
        TrackPtr track = unmatchedDetections[static_cast<size_t>(index)];
        if (track->score < args.newTrackThresh) {
            if (dbg) {
                emit(tfVf() + " NEW skip det#" + std::to_string(static_cast<int>(track->idx)) +
                     " score=" + std::to_string(track->score) + " < new_track_thresh=" +
                     std::to_string(args.newTrackThresh));
            }
            continue;
        }
        track->activate(&kalmanFilter, frameId);
        activatedStracks.push_back(track);
        noteBirth(track);
        if (dbg) {
            emit(tfVf() + " EVT NEW_TRACK id=" + std::to_string(track->trackId) +
                 " from det#" + std::to_string(static_cast<int>(track->idx)) +
                 " score=" + std::to_string(track->score) +
                 " box=" + fmtBox4(track->xyxy()) +
                 " isActivated=" + std::to_string(track->isActivated ? 1 : 0));
            checkIdBreak(track);
        }
    }

    // timeout remove on existing lost pool
    for (const TrackPtr& track : lostStracks) {
        if (frameId - track->endFrame() > maxTimeLost) {
            track->markRemoved();
            removedStracksLocal.push_back(track);
            removeReasons[track->trackId] = "timeout";
            noteDeath(track, "timeout");
            pushRecentRemoved(track, "timeout");
            if (dbg) {
                emit(tfVf() + " EVT REMOVE id=" + std::to_string(track->trackId) +
                     " reason=timeout age=" + std::to_string(frameId - track->endFrame()) +
                     " maxTimeLost=" + std::to_string(maxTimeLost) +
                     " last_score=" + std::to_string(track->score) +
                     " box=" + fmtBox4(track->xyxy()));
            }
        }
    }

    std::vector<TrackPtr> stillTracked;
    for (const TrackPtr& track : trackedStracks) {
        if (track->state == TrackState::Tracked) {
            stillTracked.push_back(track);
        }
    }
    trackedStracks = jointStracks(stillTracked, activatedStracks);
    trackedStracks = jointStracks(trackedStracks, refindStracks);
    lostStracks = subStracks(lostStracks, trackedStracks);
    lostStracks.insert(lostStracks.end(), lostStracksLocal.begin(), lostStracksLocal.end());
    // 必须同时减去历史 removed 与本帧 removedLocal，否则 timeout 轨迹会残留在 lostPool 并被重复 REMOVE
    lostStracks = subStracks(lostStracks, removedStracks);
    lostStracks = subStracks(lostStracks, removedStracksLocal);

    // 去重：debug 仅记录被剔掉的 id（不额外 markRemoved，避免改变非 debug 路径语义）
    std::set<int> beforeTrackedIds, beforeLostIds;
    if (dbg) {
        for (const auto& t : trackedStracks) beforeTrackedIds.insert(t->trackId);
        for (const auto& t : lostStracks) beforeLostIds.insert(t->trackId);
    }
    auto deduplicated = removeDuplicateStracks(trackedStracks, lostStracks);
    if (dbg) {
        std::set<int> afterIds;
        for (const auto& t : deduplicated.first) afterIds.insert(t->trackId);
        for (const auto& t : deduplicated.second) afterIds.insert(t->trackId);
        auto reportDup = [&](const std::set<int>& before, const std::vector<TrackPtr>& src) {
            for (int id : before) {
                if (afterIds.count(id)) continue;
                TrackPtr found;
                for (const auto& t : src) {
                    if (t->trackId == id) { found = t; break; }
                }
                if (!found) continue;
                removeReasons[id] = "dup";
                noteDeath(found, "dup");
                pushRecentRemoved(found, "dup");
                emit(tfVf() + " EVT REMOVE id=" + std::to_string(id) + " reason=dup box=" + fmtBox4(found->xyxy()));
            }
        };
        reportDup(beforeTrackedIds, trackedStracks);
        reportDup(beforeLostIds, lostStracks);
    }
    trackedStracks = deduplicated.first;
    lostStracks = deduplicated.second;
    removedStracks.insert(removedStracks.end(), removedStracksLocal.begin(), removedStracksLocal.end());
    if (removedStracks.size() > 1000) {
        removedStracks.erase(removedStracks.begin(), removedStracks.end() - 999);
    }

    // P2 ID_SWAP 简化：两 confirmed track 中心交叉（相对上帧）
    if (dbg && prevBoxes.size() >= 2) {
        std::vector<TrackPtr> act;
        for (const auto& t : trackedStracks) {
            if (t->isActivated && prevBoxes.count(t->trackId)) act.push_back(t);
        }
        for (size_t i = 0; i < act.size(); ++i) {
            for (size_t j = i + 1; j < act.size(); ++j) {
                const int idA = act[i]->trackId;
                const int idB = act[j]->trackId;
                const Eigen::VectorXd& pA = prevBoxes[idA];
                const Eigen::VectorXd& pB = prevBoxes[idB];
                const Eigen::VectorXd cA = act[i]->xyxy();
                const Eigen::VectorXd cB = act[j]->xyxy();
                // 若 A 更靠近 B 的旧位置，且 B 更靠近 A 的旧位置
                if (centerDistXyxy(cA, pB) + 1.0 < centerDistXyxy(cA, pA) &&
                    centerDistXyxy(cB, pA) + 1.0 < centerDistXyxy(cB, pB)) {
                    ++stats.idSwap;
                    emit(tfVf() + " ID_SWAP idA=" + std::to_string(idA) +
                         " idB=" + std::to_string(idB) +
                         " a_now=" + fmtBox4(cA) + " b_now=" + fmtBox4(cB));
                }
            }
        }
    }

    if (dbg) {
        if (!removedStracksLocal.empty()) {
            std::ostringstream rd;
            rd << tfVf() << " removed_detail:";
            for (const auto& t : removedStracksLocal) {
                const std::string reason = removeReasons.count(t->trackId) ? removeReasons[t->trackId] : "other";
                rd << " id=" << t->trackId << "(" << reason << ")";
            }
            emit(rd.str());
        }
        emit(tfVf() + " summary: activated=" +
             std::to_string(activatedStracks.size()) + " refind=" + std::to_string(refindStracks.size()) +
             " lost=" + std::to_string(lostStracksLocal.size()) + " removed=" +
             std::to_string(removedStracksLocal.size()) + " | tracked=" +
             std::to_string(trackedStracks.size()) + " lostPool=" + std::to_string(lostStracks.size()));
    }

    return resultsToMatrix(trackedStracks, fallbackResultCols);
}

void BYTETracker::logFinalStats() const {
    if (!debugCallback) return;
    std::ostringstream os;
    os << "tracker stats: total_new=" << stats.totalNew
       << " total_remove=" << stats.totalRemove
       << " total_refind=" << stats.totalRefind
       << " total_lost=" << stats.totalLost
       << " id_break_events=" << stats.idBreak
       << " id_swap_events=" << stats.idSwap
       << " lost_no_det=" << stats.lostNoDet
       << " lost_no_match=" << stats.lostNoMatch
       << " remove_timeout=" << stats.removeTimeout
       << " remove_unconfirmed=" << stats.removeUnconfirmed
       << " remove_dup=" << stats.removeDup
       << " remove_other=" << stats.removeOther
       << " lifecycle_records=" << lifecycle.size();
    emitDebug(os.str());
}

bool BYTETracker::dumpLifecycleCsv(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << "track_id,birth_vf,birth_tf,death_vf,death_tf,death_reason,num_updates,"
           "mean_cx,mean_cy,mean_w,mean_h,max_score,min_score\n";
    for (const auto& kv : lifecycle) {
        const LifecycleRec& r = kv.second;
        const double n = std::max(1, r.numUpdates);
        ofs << r.trackId << ','
            << r.birthVf << ','
            << r.birthTf << ','
            << r.deathVf << ','
            << r.deathTf << ','
            << (r.deathReason.empty() ? (r.dead ? "other" : "alive") : r.deathReason) << ','
            << r.numUpdates << ','
            << (r.sumCx / n) << ','
            << (r.sumCy / n) << ','
            << (r.sumW / n) << ','
            << (r.sumH / n) << ','
            << r.maxScore << ','
            << (r.minScore > 1e8 ? 0.0 : r.minScore) << '\n';
    }
    return true;
}

// 返回默认的 KalmanFilterXYAH 实例（供构造时初始化成员）
KalmanFilterXYAH BYTETracker::getKalmanFilter() const {
    return KalmanFilterXYAH();
}

// 将检测矩阵转为 STrack 对象列表：每行一个检测框（末尾含原始索引）
// img 参数当前保留接口兼容，未实际使用
std::vector<TrackPtr> BYTETracker::initTrack(
    const Eigen::MatrixXd& dets,
    const Eigen::VectorXd& scores,
    const Eigen::VectorXd& clsValues,
    const Eigen::MatrixXd* img
) const {
    (void)img;
    if (scores.size() != dets.rows() || clsValues.size() != dets.rows()) {
        throw std::invalid_argument("initTrack inputs must have matching row counts");
    }
    std::vector<TrackPtr> detections;
    detections.reserve(static_cast<size_t>(dets.rows()));
    for (int i = 0; i < dets.rows(); ++i) {
        detections.push_back(std::make_shared<STrack>(dets.row(i).transpose(), scores(i), clsValues(i)));
    }
    return detections;
}

// 计算轨迹与检测之间的距离矩阵：基础为 IoU 距离，若 fuseScore=true 则融合检测置信度
Eigen::MatrixXd BYTETracker::getDists(const std::vector<TrackPtr>& tracks, const std::vector<TrackPtr>& detections) const {
    Eigen::MatrixXd dists = iouDistance(tracksToBoxes(tracks), tracksToBoxes(detections));
    if (args.fuseScore) {
        Eigen::VectorXd detectionScores(static_cast<int>(detections.size()));
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            detectionScores(i) = detections[static_cast<size_t>(i)]->score;
        }
        dists = fuseScore(dists, detectionScores);
    }
    return dists;
}

// 批量预测：委托给 STrack::multiPredict
void BYTETracker::multiPredict(std::vector<TrackPtr>& tracks) const {
    STrack::multiPredict(tracks);
}

// 重置全局轨迹 ID 计数器
void BYTETracker::resetId() {
    STrack::resetId();
}

// 重置追踪器全部状态：清空所有轨迹列表、重置帧号和卡尔曼滤波器
void BYTETracker::reset() {
    trackedStracks.clear();
    lostStracks.clear();
    removedStracks.clear();
    frameId = 0;
    kalmanFilter = getKalmanFilter();
    recentRemoved.clear();
    lifecycle.clear();
    stats = DebugStats{};
    currentVideoFrameId = -1;
    initLogged = false;
    resetId();
}

// 合并两个轨迹列表，按 trackId 去重
std::vector<TrackPtr> BYTETracker::jointStracks(const std::vector<TrackPtr>& tlista, const std::vector<TrackPtr>& tlistb) {
    std::map<int, bool> exists;
    std::vector<TrackPtr> result;
    for (const TrackPtr& track : tlista) {
        exists[track->trackId] = true;
        result.push_back(track);
    }
    for (const TrackPtr& track : tlistb) {
        if (!exists[track->trackId]) {
            exists[track->trackId] = true;
            result.push_back(track);
        }
    }
    return result;
}

// 返回在 tlista 中但不在 tlistb 中的轨迹（按 trackId 差集）
std::vector<TrackPtr> BYTETracker::subStracks(const std::vector<TrackPtr>& tlista, const std::vector<TrackPtr>& tlistb) {
    std::set<int> trackIdsB;
    for (const TrackPtr& track : tlistb) {
        trackIdsB.insert(track->trackId);
    }

    std::vector<TrackPtr> result;
    for (const TrackPtr& track : tlista) {
        if (trackIdsB.find(track->trackId) == trackIdsB.end()) {
            result.push_back(track);
        }
    }
    return result;
}

// 移除 tracked 与 lost 列表中的重复轨迹：IoU < 0.15 时保留存在时间更长的那条
std::pair<std::vector<TrackPtr>, std::vector<TrackPtr>> BYTETracker::removeDuplicateStracks(
    const std::vector<TrackPtr>& stracksa,
    const std::vector<TrackPtr>& stracksb
) {
    Eigen::MatrixXd pdist = iouDistance(tracksToBoxes(stracksa), tracksToBoxes(stracksb));
    std::set<int> dupa;
    std::set<int> dupb;

    for (int p = 0; p < pdist.rows(); ++p) {
        for (int q = 0; q < pdist.cols(); ++q) {
            if (pdist(p, q) < 0.15) {
                int timep = stracksa[static_cast<size_t>(p)]->frameId - stracksa[static_cast<size_t>(p)]->startFrame;
                int timeq = stracksb[static_cast<size_t>(q)]->frameId - stracksb[static_cast<size_t>(q)]->startFrame;
                if (timep > timeq) {
                    dupb.insert(q);
                } else {
                    dupa.insert(p);
                }
            }
        }
    }

    std::vector<TrackPtr> resa;
    std::vector<TrackPtr> resb;
    for (int i = 0; i < static_cast<int>(stracksa.size()); ++i) {
        if (dupa.find(i) == dupa.end()) {
            resa.push_back(stracksa[static_cast<size_t>(i)]);
        }
    }
    for (int i = 0; i < static_cast<int>(stracksb.size()); ++i) {
        if (dupb.find(i) == dupb.end()) {
            resb.push_back(stracksb[static_cast<size_t>(i)]);
        }
    }
    return {resa, resb};
}

// 工厂函数：从默认配置文件加载参数并构造 BYTETracker
BYTETracker createByteTracker() {
    TrackerConfig config = loadTrackerConfig();
    int frameRate = config.frameRate;
    return BYTETracker(config, frameRate, /*skipN=*/1);
}

} // namespace trackerCpp
