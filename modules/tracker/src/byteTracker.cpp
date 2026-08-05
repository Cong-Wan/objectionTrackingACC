/*
Author:
Version: 1.5
Date: 2026-08-04
Description: 翻译 byte_tracker.py 的 STrack 与 BYTETracker 实现，补齐 xywh2ltwh 批量重载、消除重载歧义并增加 GMC catch-all 容错
    1.5: update() 增加 debugCallback 插桩（高/低分划分、三轮匹配代价矩阵/对数/匈牙利耗时、轨迹状态统计），
    默认关闭不影响行为。
*/

#include "byteTracker.h"

#include "matching.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace trackerCpp {
namespace {

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

// 构造追踪器：根据帧率换算最大丢失时间，并重置全局 ID
BYTETracker::BYTETracker(const TrackerConfig& argsValue, int frameRate)
    : frameId(0),
      args(argsValue),
      maxTimeLost(static_cast<int>(frameRate / 30.0 * args.trackBuffer)),
      kalmanFilter(getKalmanFilter()) {
    resetId();
}

// BYTETracker 核心更新流程：
// 1. 将检测结果按置信度分为高分/低分两组
// 2. 高分检测与已有轨迹（tracked + lost）做第一级匹配
// 3. 未匹配的 tracked 轨迹与低分检测做第二级匹配
// 4. 未确认的轨迹与剩余高分检测做第三级匹配
// 5. 仍未匹配的检测若置信度足够则新建轨迹
// 6. 更新 lost/removed 列表并去重，返回激活轨迹矩阵
Eigen::MatrixXf BYTETracker::update(const DetectionResults& results, const Eigen::MatrixXd* img, const Eigen::MatrixXd* feats) {
    Eigen::MatrixXd rawBoxes = normalizedBoxes(results);
    validateDetectionResults(results, rawBoxes);
    const int fallbackResultCols = results.hasXywhr ? 9 : 8;

    frameId += 1;

    // 调试插桩：debugCallback 非空时输出匹配代价明细（lambda 捕获，零行为变更）
    const bool dbg = static_cast<bool>(debugCallback);
    auto emit = [this](const std::string& msg) { debugCallback(msg); };
    auto nowMs = []() { return std::chrono::steady_clock::now(); };
    auto elapsedMs = [](auto t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    std::vector<TrackPtr> activatedStracks;      // 本帧新激活的轨迹
    std::vector<TrackPtr> refindStracks;         // 本帧重新找回的轨迹
    std::vector<TrackPtr> lostStracksLocal;      // 本帧新标记为丢失的轨迹
    std::vector<TrackPtr> removedStracksLocal;   // 本帧新标记为移除的轨迹

    Eigen::VectorXd scores = results.conf;
    Eigen::MatrixXd bboxes = appendIndexColumn(rawBoxes);
    Eigen::VectorXd cls = results.cls;

    // 按置信度阈值划分高分检测与低分检测的索引
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
        emit("tracker frame " + std::to_string(frameId) + ": input dets=" + std::to_string(scores.size()) +
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

    // 将高分检测初始化为 STrack 对象
    std::vector<TrackPtr> detections = initTrack(dets, scoresKeep, clsKeep, feats == nullptr ? img : feats);

    // 将已有 tracked 轨迹分为：已确认的（可参与第一级匹配）和未确认的（留到第三级匹配）
    std::vector<TrackPtr> unconfirmed;
    std::vector<TrackPtr> trackedStracksLocal;
    for (const TrackPtr& track : trackedStracks) {
        if (!track->isActivated) {
            unconfirmed.push_back(track);
        } else {
            trackedStracksLocal.push_back(track);
        }
    }

    // 构造待匹配轨迹池：已确认轨迹 + 丢失轨迹，并执行卡尔曼预测
    std::vector<TrackPtr> strackPool = jointStracks(trackedStracksLocal, lostStracks);
    multiPredict(strackPool);
    if (dbg) {
        emit("tracker frame " + std::to_string(frameId) + ": pool confirmed=" +
             std::to_string(trackedStracksLocal.size()) + " + lost=" + std::to_string(lostStracks.size()) +
             " = " + std::to_string(strackPool.size()) + ", unconfirmed=" + std::to_string(unconfirmed.size()));
    }

    // 若提供了 GMC 回调且存在图像数据，则执行全局运动补偿；异常时回退到单位变换
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

    // 第一级匹配：strackPool（tracked + lost） vs 高分检测
    auto tR1 = nowMs();
    Eigen::MatrixXd dists = getDists(strackPool, detections);
    AssignmentResult assignment = linearAssignment(dists, args.matchThresh);
    if (dbg) {
        emit("tracker frame " + std::to_string(frameId) + " round1: cost " +
             std::to_string(strackPool.size()) + "x" + std::to_string(detections.size()) +
             " (iouDist+score, thresh=" + std::to_string(args.matchThresh) + "), hungarian=" +
             std::to_string(elapsedMs(tR1)) + " ms, matches=" + std::to_string(assignment.matches.size()) +
             " unmatchedTracks=" + std::to_string(assignment.unmatchedA.size()) +
             " unmatchedDets=" + std::to_string(assignment.unmatchedB.size()));
        for (const auto& m : assignment.matches) {
            const TrackPtr& det = detections[static_cast<size_t>(m.second)];
            emit("  round1 match: trackId=" + std::to_string(strackPool[static_cast<size_t>(m.first)]->trackId) +
                 " <- det#" + std::to_string(static_cast<int>(det->idx)) +
                 " score=" + std::to_string(det->score) +
                 " cost=" + std::to_string(dists(m.first, m.second)));
        }
    }

    for (const auto& match : assignment.matches) {
        TrackPtr track = strackPool[static_cast<size_t>(match.first)];
        const TrackPtr& det = detections[static_cast<size_t>(match.second)];
        if (track->state == TrackState::Tracked) {
            track->update(*det, frameId);
            activatedStracks.push_back(track);
        } else {
            track->reActivate(*det, frameId, false);
            refindStracks.push_back(track);
        }
    }

    // 第二级匹配：第一级未匹配的 tracked 轨迹 vs 低分检测
    std::vector<TrackPtr> detectionsSecond = initTrack(detsSecond, scoresSecond, clsSecond, feats == nullptr ? img : feats);
    std::vector<TrackPtr> rTrackedStracks;
    for (int index : assignment.unmatchedA) {
        if (strackPool[static_cast<size_t>(index)]->state == TrackState::Tracked) {
            rTrackedStracks.push_back(strackPool[static_cast<size_t>(index)]);
        }
    }

    auto tR2 = nowMs();
    dists = iouDistance(tracksToBoxes(rTrackedStracks), tracksToBoxes(detectionsSecond));
    AssignmentResult secondAssignment = linearAssignment(dists, 0.5);
    if (dbg) {
        emit("tracker frame " + std::to_string(frameId) + " round2: cost " +
             std::to_string(rTrackedStracks.size()) + "x" + std::to_string(detectionsSecond.size()) +
             " (iouDist, thresh=0.5), hungarian=" + std::to_string(elapsedMs(tR2)) +
             " ms, matches=" + std::to_string(secondAssignment.matches.size()) +
             " unmatchedTracks(lost)=" + std::to_string(secondAssignment.unmatchedA.size()) +
             " unmatchedLowDets=" + std::to_string(secondAssignment.unmatchedB.size()));
        for (const auto& m : secondAssignment.matches) {
            const TrackPtr& det = detectionsSecond[static_cast<size_t>(m.second)];
            emit("  round2 match: trackId=" + std::to_string(rTrackedStracks[static_cast<size_t>(m.first)]->trackId) +
                 " <- lowDet#" + std::to_string(static_cast<int>(det->idx)) +
                 " score=" + std::to_string(det->score) +
                 " cost=" + std::to_string(dists(m.first, m.second)));
        }
    }
    for (const auto& match : secondAssignment.matches) {
        TrackPtr track = rTrackedStracks[static_cast<size_t>(match.first)];
        const TrackPtr& det = detectionsSecond[static_cast<size_t>(match.second)];
        if (track->state == TrackState::Tracked) {
            track->update(*det, frameId);
            activatedStracks.push_back(track);
        } else {
            track->reActivate(*det, frameId, false);
            refindStracks.push_back(track);
        }
    }

    // 第二级仍未匹配的 tracked 轨迹标记为丢失
    for (int index : secondAssignment.unmatchedA) {
        TrackPtr track = rTrackedStracks[static_cast<size_t>(index)];
        if (track->state != TrackState::Lost) {
            track->markLost();
            lostStracksLocal.push_back(track);
        }
    }

    // 第三级匹配：未确认轨迹 vs 第一级未匹配的高分检测
    std::vector<TrackPtr> unmatchedDetections = selectTracks(detections, assignment.unmatchedB);
    auto tR3 = nowMs();
    dists = getDists(unconfirmed, unmatchedDetections);
    AssignmentResult unconfirmedAssignment = linearAssignment(dists, 0.7);
    if (dbg) {
        emit("tracker frame " + std::to_string(frameId) + " round3: cost " +
             std::to_string(unconfirmed.size()) + "x" + std::to_string(unmatchedDetections.size()) +
             " (iouDist, thresh=0.7), hungarian=" + std::to_string(elapsedMs(tR3)) +
             " ms, matches=" + std::to_string(unconfirmedAssignment.matches.size()) +
             " unmatchedUnconfirmed(removed)=" + std::to_string(unconfirmedAssignment.unmatchedA.size()) +
             " unmatchedDets=" + std::to_string(unconfirmedAssignment.unmatchedB.size()));
    }
    for (const auto& match : unconfirmedAssignment.matches) {
        TrackPtr track = unconfirmed[static_cast<size_t>(match.first)];
        track->update(*unmatchedDetections[static_cast<size_t>(match.second)], frameId);
        activatedStracks.push_back(track);
    }

    // 第三级未匹配的未确认轨迹直接移除
    for (int index : unconfirmedAssignment.unmatchedA) {
        TrackPtr track = unconfirmed[static_cast<size_t>(index)];
        track->markRemoved();
        removedStracksLocal.push_back(track);
    }

    // 第三级未匹配且置信度足够的高分检测新建轨迹
    for (int index : unconfirmedAssignment.unmatchedB) {
        TrackPtr track = unmatchedDetections[static_cast<size_t>(index)];
        if (track->score < args.newTrackThresh) {
            continue;
        }
        track->activate(&kalmanFilter, frameId);
        activatedStracks.push_back(track);
    }

    // 清理长时间未更新的丢失轨迹（超过 maxTimeLost 帧）
    for (const TrackPtr& track : lostStracks) {
        if (frameId - track->endFrame() > maxTimeLost) {
            track->markRemoved();
            removedStracksLocal.push_back(track);
        }
    }

    // 汇总本帧状态：保留仍在追踪的轨迹，合并新激活/找回的轨迹，更新 lost/removed 列表并去重
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
    lostStracks = subStracks(lostStracks, removedStracks);

    // 移除 tracked 与 lost 之间的重复轨迹（IoU < 0.15 视为重复）
    auto deduplicated = removeDuplicateStracks(trackedStracks, lostStracks);
    trackedStracks = deduplicated.first;
    lostStracks = deduplicated.second;
    removedStracks.insert(removedStracks.end(), removedStracksLocal.begin(), removedStracksLocal.end());
    // 防止 removedStracks 无限增长，仅保留最近 999 条
    if (removedStracks.size() > 1000) {
        removedStracks.erase(removedStracks.begin(), removedStracks.end() - 999);
    }

    if (dbg) {
        emit("tracker frame " + std::to_string(frameId) + " summary: activated=" +
             std::to_string(activatedStracks.size()) + " refind=" + std::to_string(refindStracks.size()) +
             " lost=" + std::to_string(lostStracksLocal.size()) + " removed=" +
             std::to_string(removedStracksLocal.size()) + " | tracked=" +
             std::to_string(trackedStracks.size()) + " lostPool=" + std::to_string(lostStracks.size()));
    }

    return resultsToMatrix(trackedStracks, fallbackResultCols);
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
    return BYTETracker(config, frameRate);
}

} // namespace trackerCpp
