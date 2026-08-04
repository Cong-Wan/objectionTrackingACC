/*
Author:
Version: 1.0
Date: 2026-06-06
Description: 翻译 utils/matching.py 的匹配、IoU、线性分配函数声明
*/

#ifndef TRACKER_CPP_MATCHING_H
#define TRACKER_CPP_MATCHING_H

#include <Eigen/Dense>
#include <string>
#include <utility>
#include <vector>

namespace trackerCpp {

// 线性分配结果：成功匹配对 + 两侧未匹配的索引
struct AssignmentResult {
    std::vector<std::pair<int, int>> matches;  // (A索引, B索引) 的匹配对列表
    std::vector<int> unmatchedA;               // A 侧未匹配的索引
    std::vector<int> unmatchedB;               // B 侧未匹配的索引
};

// 计算两组轴对齐框的 IoA/IoU 矩阵
// iou=false 时计算 IoA(Intersection over Area)，否则计算 IoU
Eigen::MatrixXd bboxIoa(
    const Eigen::MatrixXd& box1,
    const Eigen::MatrixXd& box2,
    bool iou = false,
    double eps = 1e-7
);

// 批量计算旋转框(obb)之间的概率 IoU，输入为 [N,5] (x,y,w,h,angle)
Eigen::MatrixXd batchProbIou(
    const Eigen::MatrixXd& obb1,
    const Eigen::MatrixXd& obb2,
    double eps = 1e-7
);

// 线性分配（匈牙利算法）：在代价矩阵中寻找最优匹配
// thresh 为允许的最大代价，超过此值的匹配会被丢弃
// useLap=true 使用 LAP 扩展方阵策略，否则使用纯匈牙利 fallback
AssignmentResult linearAssignment(
    const Eigen::MatrixXd& costMatrix,
    double thresh,
    bool useLap = true
);

// 将 IoU/PIoU 转换为距离矩阵：distance = 1 - iou
// 支持 4 列 (xyxy) 和 5 列 (xywha) 两种格式
Eigen::MatrixXd iouDistance(
    const Eigen::MatrixXd& aTlbrs,
    const Eigen::MatrixXd& bTlbrs
);

// 计算外观特征距离矩阵，metric 支持 "cosine" 和 "euclidean"
Eigen::MatrixXd embeddingDistance(
    const Eigen::MatrixXd& trackFeatures,
    const Eigen::MatrixXd& detectionFeatures,
    const std::string& metric = "cosine"
);

// 将 IoU 代价矩阵与检测置信度融合：cost = 1 - (1 - iouCost) * score
Eigen::MatrixXd fuseScore(
    const Eigen::MatrixXd& costMatrix,
    const Eigen::VectorXd& detectionScores
);

} // namespace trackerCpp

#endif // TRACKER_CPP_MATCHING_H
