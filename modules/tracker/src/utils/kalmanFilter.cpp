/*
Author:
Version: 1.0
Date: 2026-06-06
Description: 翻译 utils/kalman_filter.py 的 KalmanFilterXYAH 与 KalmanFilterXYWH 实现
*/

#include "kalmanFilter.h"

#include <cmath>
#include <stdexcept>

namespace trackerCpp {
namespace {

// 将向量的每个元素平方后构成对角矩阵
Eigen::MatrixXd diagonalSquared(const Eigen::VectorXd& values) {
    return values.array().square().matrix().asDiagonal();
}

} // namespace

KalmanFilterXYAH::KalmanFilterXYAH()
    : motionMat(Eigen::MatrixXd::Identity(8, 8)),
      updateMat(Eigen::MatrixXd::Zero(4, 8)),
      stdWeightPosition(1.0 / 20.0),
      stdWeightVelocity(1.0 / 160.0) {
    const double dt = 1.0;
    // 构造匀速运动模型：状态 [x,y,a,h,vx,vy,va,vh]
    // motionMat 将位置更新为 position + velocity * dt
    for (int i = 0; i < 4; ++i) {
        motionMat(i, 4 + i) = dt;
        updateMat(i, i) = 1.0;
    }
}

// 根据首次观测（xyah）初始化状态均值与协方差
// 位置部分取观测值，速度部分置零；协方差按经验公式设定
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYAH::initiate(const Eigen::VectorXd& measurement) const {
    Eigen::VectorXd mean(8);
    mean.head(4) = measurement;
    mean.tail(4).setZero();

    Eigen::VectorXd std(8);
    std << 2.0 * stdWeightPosition * measurement(3),
        2.0 * stdWeightPosition * measurement(3),
        1e-2,
        2.0 * stdWeightPosition * measurement(3),
        10.0 * stdWeightVelocity * measurement(3),
        10.0 * stdWeightVelocity * measurement(3),
        1e-5,
        10.0 * stdWeightVelocity * measurement(3);

    return {mean, diagonalSquared(std)};
}

// 预测步骤：x' = F*x, P' = F*P*F^T + Q
// 过程噪声 Q 与目标高度相关，用于适应尺度变化
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYAH::predict(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance
) const {
    Eigen::VectorXd std(8);
    std << stdWeightPosition * mean(3),
        stdWeightPosition * mean(3),
        1e-2,
        stdWeightPosition * mean(3),
        stdWeightVelocity * mean(3),
        stdWeightVelocity * mean(3),
        1e-5,
        stdWeightVelocity * mean(3);

    Eigen::MatrixXd motionCov = diagonalSquared(std);
    Eigen::VectorXd predictedMean = motionMat * mean;
    Eigen::MatrixXd predictedCovariance = motionMat * covariance * motionMat.transpose() + motionCov;
    return {predictedMean, predictedCovariance};
}

// 将状态空间投影到观测空间：z = H*x, S = H*P*H^T + R
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYAH::project(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance
) const {
    Eigen::VectorXd std(4);
    std << stdWeightPosition * mean(3),
        stdWeightPosition * mean(3),
        1e-1,
        stdWeightPosition * mean(3);

    Eigen::MatrixXd innovationCov = diagonalSquared(std);
    Eigen::VectorXd projectedMean = updateMat * mean;
    Eigen::MatrixXd projectedCovariance = updateMat * covariance * updateMat.transpose();
    return {projectedMean, projectedCovariance + innovationCov};
}

// 对多条轨迹批量执行 predict，输入为状态矩阵 [N,8] 和协方差列表
std::pair<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> KalmanFilterXYAH::multiPredict(
    const Eigen::MatrixXd& mean,
    const std::vector<Eigen::MatrixXd>& covariance
) const {
    Eigen::MatrixXd predictedMean(mean.rows(), mean.cols());
    std::vector<Eigen::MatrixXd> predictedCovariance;
    predictedCovariance.reserve(covariance.size());

    for (int i = 0; i < mean.rows(); ++i) {
        auto predicted = predict(mean.row(i).transpose(), covariance[static_cast<size_t>(i)]);
        predictedMean.row(i) = predicted.first.transpose();
        predictedCovariance.push_back(predicted.second);
    }

    return {predictedMean, predictedCovariance};
}

// 更新步骤：K = P*H^T*S^{-1}, x = x + K*(z - H*x), P = P - K*S*K^T
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYAH::update(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const Eigen::VectorXd& measurement
) const {
    auto projected = project(mean, covariance);
    const Eigen::VectorXd& projectedMean = projected.first;
    const Eigen::MatrixXd& projectedCovariance = projected.second;

    Eigen::MatrixXd gainRight = (covariance * updateMat.transpose()).transpose();
    Eigen::MatrixXd kalmanGain = projectedCovariance.llt().solve(gainRight).transpose();
    Eigen::VectorXd innovation = measurement - projectedMean;

    Eigen::VectorXd newMean = mean + kalmanGain * innovation;
    Eigen::MatrixXd newCovariance = covariance - kalmanGain * projectedCovariance * kalmanGain.transpose();
    return {newMean, newCovariance};
}

// 计算多个观测与当前预测状态的门控距离
// onlyPosition=true 时仅比较 x,y 分量；metric="maha" 使用马氏距离，"gaussian" 使用欧氏距离
Eigen::VectorXd KalmanFilterXYAH::gatingDistance(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const Eigen::MatrixXd& measurements,
    bool onlyPosition,
    const std::string& metric
) const {
    auto projected = project(mean, covariance);
    Eigen::VectorXd projectedMean = projected.first;
    Eigen::MatrixXd projectedCovariance = projected.second;
    Eigen::MatrixXd projectedMeasurements = measurements;

    if (onlyPosition) {
        projectedMean = projectedMean.head(2);
        projectedCovariance = projectedCovariance.topLeftCorner(2, 2);
        projectedMeasurements = measurements.leftCols(2);
    }

    Eigen::VectorXd distances(projectedMeasurements.rows());
    for (int i = 0; i < projectedMeasurements.rows(); ++i) {
        Eigen::VectorXd d = projectedMeasurements.row(i).transpose() - projectedMean;
        if (metric == "gaussian") {
            distances(i) = d.squaredNorm();
        } else if (metric == "maha") {
            Eigen::VectorXd z = projectedCovariance.llt().matrixL().solve(d);
            distances(i) = z.squaredNorm();
        } else {
            throw std::invalid_argument("Invalid distance metric");
        }
    }

    return distances;
}

// XYWH 版本的初始化：协方差与宽度、高度均相关（区别于 XYAH 仅与高度相关）
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYWH::initiate(const Eigen::VectorXd& measurement) const {
    Eigen::VectorXd mean(8);
    mean.head(4) = measurement;
    mean.tail(4).setZero();

    Eigen::VectorXd std(8);
    std << 2.0 * stdWeightPosition * measurement(2),
        2.0 * stdWeightPosition * measurement(3),
        2.0 * stdWeightPosition * measurement(2),
        2.0 * stdWeightPosition * measurement(3),
        10.0 * stdWeightVelocity * measurement(2),
        10.0 * stdWeightVelocity * measurement(3),
        10.0 * stdWeightVelocity * measurement(2),
        10.0 * stdWeightVelocity * measurement(3);

    return {mean, diagonalSquared(std)};
}

// XYWH 版本的预测：过程噪声与宽、高均相关
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYWH::predict(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance
) const {
    Eigen::VectorXd std(8);
    std << stdWeightPosition * mean(2),
        stdWeightPosition * mean(3),
        stdWeightPosition * mean(2),
        stdWeightPosition * mean(3),
        stdWeightVelocity * mean(2),
        stdWeightVelocity * mean(3),
        stdWeightVelocity * mean(2),
        stdWeightVelocity * mean(3);

    Eigen::MatrixXd motionCov = diagonalSquared(std);
    Eigen::VectorXd predictedMean = motionMat * mean;
    Eigen::MatrixXd predictedCovariance = motionMat * covariance * motionMat.transpose() + motionCov;
    return {predictedMean, predictedCovariance};
}

// XYWH 版本的投影：观测噪声与宽、高均相关
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYWH::project(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance
) const {
    Eigen::VectorXd std(4);
    std << stdWeightPosition * mean(2),
        stdWeightPosition * mean(3),
        stdWeightPosition * mean(2),
        stdWeightPosition * mean(3);

    Eigen::MatrixXd innovationCov = diagonalSquared(std);
    Eigen::VectorXd projectedMean = updateMat * mean;
    Eigen::MatrixXd projectedCovariance = updateMat * covariance * updateMat.transpose();
    return {projectedMean, projectedCovariance + innovationCov};
}

// XYWH 版本的批量预测，逻辑同 KalmanFilterXYAH::multiPredict
std::pair<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> KalmanFilterXYWH::multiPredict(
    const Eigen::MatrixXd& mean,
    const std::vector<Eigen::MatrixXd>& covariance
) const {
    Eigen::MatrixXd predictedMean(mean.rows(), mean.cols());
    std::vector<Eigen::MatrixXd> predictedCovariance;
    predictedCovariance.reserve(covariance.size());

    for (int i = 0; i < mean.rows(); ++i) {
        auto predicted = predict(mean.row(i).transpose(), covariance[static_cast<size_t>(i)]);
        predictedMean.row(i) = predicted.first.transpose();
        predictedCovariance.push_back(predicted.second);
    }

    return {predictedMean, predictedCovariance};
}

// XYWH 版本的更新步骤直接复用父类实现（观测投影维度相同）
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilterXYWH::update(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const Eigen::VectorXd& measurement
) const {
    return KalmanFilterXYAH::update(mean, covariance, measurement);
}

} // namespace trackerCpp
