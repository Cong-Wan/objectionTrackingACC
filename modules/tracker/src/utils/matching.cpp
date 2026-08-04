/*
Author:
Version: 1.2
Date: 2026-06-06
Description: 翻译 utils/matching.py 的匹配、IoU、线性分配函数实现，补齐 useLap=false 分支并严格校验 bbox 列数
*/

#include "matching.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace trackerCpp {
namespace {

// 生成 0..size-1 的连续整数序列，用于初始化未匹配索引
std::vector<int> makeRange(int size) {
    std::vector<int> values(static_cast<size_t>(size));
    std::iota(values.begin(), values.end(), 0);
    return values;
}

// 匈牙利算法（行优先实现）：求解矩形代价矩阵的最小权匹配
// 返回长度为 rows 的向量，assignment[i] 表示第 i 行匹配的列号，-1 表示未匹配
std::vector<int> hungarianForRows(const Eigen::MatrixXd& cost) {
    const int rows = static_cast<int>(cost.rows());
    const int cols = static_cast<int>(cost.cols());
    const double inf = std::numeric_limits<double>::infinity();

    std::vector<double> u(static_cast<size_t>(rows + 1), 0.0);
    std::vector<double> v(static_cast<size_t>(cols + 1), 0.0);
    std::vector<int> p(static_cast<size_t>(cols + 1), 0);
    std::vector<int> way(static_cast<size_t>(cols + 1), 0);

    for (int i = 1; i <= rows; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(static_cast<size_t>(cols + 1), inf);
        std::vector<char> used(static_cast<size_t>(cols + 1), false);

        do {
            used[static_cast<size_t>(j0)] = true;
            int i0 = p[static_cast<size_t>(j0)];
            double delta = inf;
            int j1 = 0;

            for (int j = 1; j <= cols; ++j) {
                if (used[static_cast<size_t>(j)]) {
                    continue;
                }
                double current = cost(i0 - 1, j - 1) - u[static_cast<size_t>(i0)] - v[static_cast<size_t>(j)];
                if (current < minv[static_cast<size_t>(j)]) {
                    minv[static_cast<size_t>(j)] = current;
                    way[static_cast<size_t>(j)] = j0;
                }
                if (minv[static_cast<size_t>(j)] < delta) {
                    delta = minv[static_cast<size_t>(j)];
                    j1 = j;
                }
            }

            for (int j = 0; j <= cols; ++j) {
                if (used[static_cast<size_t>(j)]) {
                    u[static_cast<size_t>(p[static_cast<size_t>(j)])] += delta;
                    v[static_cast<size_t>(j)] -= delta;
                } else {
                    minv[static_cast<size_t>(j)] -= delta;
                }
            }
            j0 = j1;
        } while (p[static_cast<size_t>(j0)] != 0);

        do {
            int j1 = way[static_cast<size_t>(j0)];
            p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment(static_cast<size_t>(rows), -1);
    for (int j = 1; j <= cols; ++j) {
        if (p[static_cast<size_t>(j)] > 0) {
            assignment[static_cast<size_t>(p[static_cast<size_t>(j)] - 1)] = j - 1;
        }
    }
    return assignment;
}

// 校验矩阵列数，不匹配时抛出异常
void requireColumnsEqual(const Eigen::MatrixXd& matrix, int cols, const std::string& name) {
    if (matrix.cols() != cols) {
        throw std::invalid_argument(name + " must have " + std::to_string(cols) + " columns");
    }
}

// 将匈牙利算法的行/列分配结果统一转为 (row, col) 配对列表
std::vector<std::pair<int, int>> hungarianPairs(const Eigen::MatrixXd& cost) {
    std::vector<std::pair<int, int>> pairs;
    // 行数 <= 列数时直接对行做分配；否则转置后对列做分配再映射回来
    if (cost.rows() <= cost.cols()) {
        std::vector<int> assignment = hungarianForRows(cost);
        for (int row = 0; row < static_cast<int>(assignment.size()); ++row) {
            int col = assignment[static_cast<size_t>(row)];
            if (col >= 0) {
                pairs.emplace_back(row, col);
            }
        }
    } else {
        Eigen::MatrixXd transposed = cost.transpose();
        std::vector<int> assignment = hungarianForRows(transposed);
        for (int col = 0; col < static_cast<int>(assignment.size()); ++col) {
            int row = assignment[static_cast<size_t>(col)];
            if (row >= 0) {
                pairs.emplace_back(row, col);
            }
        }
    }
    return pairs;
}

// 不使用 LAP 扩展方阵的 fallback 分配策略：直接对原始代价矩阵做匈牙利分配，然后按阈值过滤
AssignmentResult linearAssignmentScipyFallback(const Eigen::MatrixXd& costMatrix, double thresh) {
    AssignmentResult result;
    std::vector<std::pair<int, int>> pairs = hungarianPairs(costMatrix);
    std::set<int> matchedA;
    std::set<int> matchedB;

    for (const auto& pair : pairs) {
        if (costMatrix(pair.first, pair.second) <= thresh) {
            result.matches.push_back(pair);
            matchedA.insert(pair.first);
            matchedB.insert(pair.second);
        }
    }

    for (int row = 0; row < costMatrix.rows(); ++row) {
        if (matchedA.find(row) == matchedA.end()) {
            result.unmatchedA.push_back(row);
        }
    }
    for (int col = 0; col < costMatrix.cols(); ++col) {
        if (matchedB.find(col) == matchedB.end()) {
            result.unmatchedB.push_back(col);
        }
    }
    return result;
}

} // namespace

// 计算两组轴对齐框两两之间的 IoA 或 IoU
// box1: [N,4] (x1,y1,x2,y2), box2: [M,4]，返回 [N,M] 结果矩阵
Eigen::MatrixXd bboxIoa(const Eigen::MatrixXd& box1, const Eigen::MatrixXd& box2, bool iou, double eps) {
    requireColumnsEqual(box1, 4, "box1");
    requireColumnsEqual(box2, 4, "box2");
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(box1.rows(), box2.rows());

    for (int i = 0; i < box1.rows(); ++i) {
        const double b1X1 = box1(i, 0);
        const double b1Y1 = box1(i, 1);
        const double b1X2 = box1(i, 2);
        const double b1Y2 = box1(i, 3);
        const double box1Area = (b1X2 - b1X1) * (b1Y2 - b1Y1);

        for (int j = 0; j < box2.rows(); ++j) {
            const double b2X1 = box2(j, 0);
            const double b2Y1 = box2(j, 1);
            const double b2X2 = box2(j, 2);
            const double b2Y2 = box2(j, 3);

            const double interWidth = std::max(0.0, std::min(b1X2, b2X2) - std::max(b1X1, b2X1));
            const double interHeight = std::max(0.0, std::min(b1Y2, b2Y2) - std::max(b1Y1, b2Y1));
            const double interArea = interWidth * interHeight;
            double area = (b2X2 - b2X1) * (b2Y2 - b2Y1);
            if (iou) {
                area = area + box1Area - interArea;
            }
            result(i, j) = interArea / (area + eps);
        }
    }

    return result;
}

// 批量计算旋转框(obb)之间的概率 IoU
// 基于高斯分布建模，输入 [N,5] 和 [M,5] (x,y,w,h,angle)，返回 [N,M] 的 1 - hd 矩阵
Eigen::MatrixXd batchProbIou(const Eigen::MatrixXd& obb1, const Eigen::MatrixXd& obb2, double eps) {
    requireColumnsEqual(obb1, 5, "obb1");
    requireColumnsEqual(obb2, 5, "obb2");
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(obb1.rows(), obb2.rows());

    for (int i = 0; i < obb1.rows(); ++i) {
        const double x1 = obb1(i, 0);
        const double y1 = obb1(i, 1);
        const double w1 = obb1(i, 2);
        const double h1 = obb1(i, 3);
        const double r1 = obb1(i, 4);
        const double aw1 = w1 * w1 / 12.0;
        const double bh1 = h1 * h1 / 12.0;
        const double cos1 = std::cos(r1);
        const double sin1 = std::sin(r1);
        const double a1 = aw1 * cos1 * cos1 + bh1 * sin1 * sin1;
        const double b1 = aw1 * sin1 * sin1 + bh1 * cos1 * cos1;
        const double c1 = (aw1 - bh1) * cos1 * sin1;

        for (int j = 0; j < obb2.rows(); ++j) {
            const double x2 = obb2(j, 0);
            const double y2 = obb2(j, 1);
            const double w2 = obb2(j, 2);
            const double h2 = obb2(j, 3);
            const double r2 = obb2(j, 4);
            const double aw2 = w2 * w2 / 12.0;
            const double bh2 = h2 * h2 / 12.0;
            const double cos2 = std::cos(r2);
            const double sin2 = std::sin(r2);
            const double a2 = aw2 * cos2 * cos2 + bh2 * sin2 * sin2;
            const double b2 = aw2 * sin2 * sin2 + bh2 * cos2 * cos2;
            const double c2 = (aw2 - bh2) * cos2 * sin2;

            const double denominator = (a1 + a2) * (b1 + b2) - (c1 + c2) * (c1 + c2) + eps;
            const double t1 = (((a1 + a2) * std::pow(y1 - y2, 2.0) + (b1 + b2) * std::pow(x1 - x2, 2.0)) / denominator) * 0.25;
            const double t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denominator) * 0.5;
            const double det1 = std::max(0.0, a1 * b1 - c1 * c1);
            const double det2 = std::max(0.0, a2 * b2 - c2 * c2);
            const double t3 = std::log(((a1 + a2) * (b1 + b2) - (c1 + c2) * (c1 + c2)) /
                                       (4.0 * std::sqrt(det1 * det2) + eps) + eps) * 0.5;
            const double bd = std::clamp(t1 + t2 + t3, eps, 100.0);
            const double hd = std::sqrt(1.0 - std::exp(-bd) + eps);
            result(i, j) = 1.0 - hd;
        }
    }

    return result;
}

// 线性分配主入口：根据 useLap 标志选择 LAP 扩展方阵策略或纯匈牙利 fallback
// thresh 为允许的最大代价，超过此值的匹配会被丢弃
AssignmentResult linearAssignment(const Eigen::MatrixXd& costMatrix, double thresh, bool useLap) {
    AssignmentResult result;
    const int rows = static_cast<int>(costMatrix.rows());
    const int cols = static_cast<int>(costMatrix.cols());

    // 空矩阵直接返回全部未匹配
    if (costMatrix.size() == 0) {
        result.unmatchedA = makeRange(rows);
        result.unmatchedB = makeRange(cols);
        return result;
    }
    if (!std::isfinite(thresh)) {
        throw std::invalid_argument("linearAssignment threshold must be finite");
    }

    double maxFiniteCost = 0.0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double value = costMatrix(row, col);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("linearAssignment cost matrix must be finite");
            }
            maxFiniteCost = std::max(maxFiniteCost, std::abs(value));
        }
    }
    if (!useLap) {
        return linearAssignmentScipyFallback(costMatrix, thresh);
    }

    // LAP 策略：构造 (rows+cols) x (rows+cols) 的扩展方阵
    // 原始有效区域保留实际代价；补充区域填入 unmatchedCost；右下角填 0
    const int size = rows + cols;
    const double unmatchedCost = thresh + 1e-6;
    const double invalidCost = std::max({maxFiniteCost + std::abs(thresh) + 1.0, std::abs(unmatchedCost) * 4.0 + 1.0, 1e6});
    Eigen::MatrixXd extended = Eigen::MatrixXd::Constant(size, size, invalidCost);

    // 填充扩展矩阵：左上角为实际代价（仅保留 <= thresh 的项）
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (costMatrix(row, col) <= thresh) {
                extended(row, col) = costMatrix(row, col);
            }
        }
    }
    // 右侧填充块：行 -> 虚拟列，表示未匹配惩罚
    for (int row = 0; row < rows; ++row) {
        for (int col = cols; col < size; ++col) {
            extended(row, col) = unmatchedCost;
        }
    }
    // 下侧填充块：虚拟行 -> 列，表示未匹配惩罚
    for (int row = rows; row < size; ++row) {
        for (int col = 0; col < cols; ++col) {
            extended(row, col) = unmatchedCost;
        }
    }
    // 右下角填充 0，保证方阵可解
    for (int row = rows; row < size; ++row) {
        for (int col = cols; col < size; ++col) {
            extended(row, col) = 0.0;
        }
    }

    std::vector<int> assignment = hungarianForRows(extended);
    std::set<int> matchedA;
    std::set<int> matchedB;
    // 从扩展方阵的分配结果中提取原始 rows x cols 范围内的有效匹配
    for (int row = 0; row < rows; ++row) {
        int col = assignment[static_cast<size_t>(row)];
        if (col >= 0 && col < cols && costMatrix(row, col) <= thresh) {
            result.matches.emplace_back(row, col);
            matchedA.insert(row);
            matchedB.insert(col);
        }
    }

    for (int row = 0; row < rows; ++row) {
        if (matchedA.find(row) == matchedA.end()) {
            result.unmatchedA.push_back(row);
        }
    }
    for (int col = 0; col < cols; ++col) {
        if (matchedB.find(col) == matchedB.end()) {
            result.unmatchedB.push_back(col);
        }
    }

    return result;
}

// 将包围框转换为 IoU 距离矩阵：distance = 1 - IoU
// 支持 4 列 (xyxy) 和 5 列 (xywha) 两种格式
Eigen::MatrixXd iouDistance(const Eigen::MatrixXd& aTlbrs, const Eigen::MatrixXd& bTlbrs) {
    if (aTlbrs.rows() == 0 || bTlbrs.rows() == 0) {
        return Eigen::MatrixXd::Zero(aTlbrs.rows(), bTlbrs.rows());
    }
    if (aTlbrs.cols() != bTlbrs.cols()) {
        throw std::invalid_argument("iouDistance inputs must use the same box format");
    }

    Eigen::MatrixXd ious;
    if (aTlbrs.cols() == 5) {
        ious = batchProbIou(aTlbrs, bTlbrs);
    } else if (aTlbrs.cols() == 4) {
        ious = bboxIoa(aTlbrs, bTlbrs, true);
    } else {
        throw std::invalid_argument("iouDistance boxes must have 4 or 5 columns");
    }
    return Eigen::MatrixXd::Ones(ious.rows(), ious.cols()) - ious;
}

// 计算轨迹与检测之间的外观特征距离矩阵
// metric="cosine" 时为余弦距离(0~2)，"euclidean" 时为欧氏距离
Eigen::MatrixXd embeddingDistance(
    const Eigen::MatrixXd& trackFeatures,
    const Eigen::MatrixXd& detectionFeatures,
    const std::string& metric
) {
    if (trackFeatures.cols() != detectionFeatures.cols()) {
        throw std::invalid_argument("embedding feature dimensions must match");
    }
    Eigen::MatrixXd costMatrix = Eigen::MatrixXd::Zero(trackFeatures.rows(), detectionFeatures.rows());
    if (costMatrix.size() == 0) {
        return costMatrix;
    }

    for (int i = 0; i < trackFeatures.rows(); ++i) {
        for (int j = 0; j < detectionFeatures.rows(); ++j) {
            Eigen::VectorXd diff = trackFeatures.row(i).transpose() - detectionFeatures.row(j).transpose();
            if (metric == "cosine") {
                const double denom = trackFeatures.row(i).norm() * detectionFeatures.row(j).norm();
                const double value = denom > 0.0 ? 1.0 - trackFeatures.row(i).dot(detectionFeatures.row(j)) / denom : 1.0;
                costMatrix(i, j) = std::max(0.0, value);
            } else if (metric == "euclidean") {
                costMatrix(i, j) = std::max(0.0, diff.norm());
            } else {
                throw std::invalid_argument("Unsupported embedding distance metric");
            }
        }
    }

    return costMatrix;
}

// 将 IoU 代价矩阵与检测置信度融合：
// 先转成相似度 sim = 1 - cost，再乘以 detectionScores，最后转回代价 cost = 1 - fusedSim
Eigen::MatrixXd fuseScore(const Eigen::MatrixXd& costMatrix, const Eigen::VectorXd& detectionScores) {
    if (costMatrix.size() == 0) {
        return costMatrix;
    }
    if (detectionScores.size() != costMatrix.cols()) {
        throw std::invalid_argument("detectionScores size must match costMatrix columns");
    }

    Eigen::MatrixXd iouSim = Eigen::MatrixXd::Ones(costMatrix.rows(), costMatrix.cols()) - costMatrix;
    Eigen::MatrixXd fused = iouSim;
    for (int row = 0; row < fused.rows(); ++row) {
        for (int col = 0; col < fused.cols(); ++col) {
            fused(row, col) *= detectionScores(col);
        }
    }
    return Eigen::MatrixXd::Ones(fused.rows(), fused.cols()) - fused;
}

} // namespace trackerCpp
