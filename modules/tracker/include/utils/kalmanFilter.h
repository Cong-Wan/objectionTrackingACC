/*
Author:
Version: 1.0
Date: 2026-06-06
Description: 翻译 utils/kalman_filter.py 的 KalmanFilterXYAH 与 KalmanFilterXYWH 声明
*/

#ifndef TRACKER_CPP_KALMAN_FILTER_H
#define TRACKER_CPP_KALMAN_FILTER_H

#include <Eigen/Dense>
#include <string>
#include <utility>
#include <vector>

namespace trackerCpp {

// 基于 xyah（中心x, 中心y, 宽高比, 高度）观测的卡尔曼滤波器
class KalmanFilterXYAH {
public:
    KalmanFilterXYAH();
    virtual ~KalmanFilterXYAH() = default;

    // 根据首次观测初始化卡尔曼状态（均值与协方差）
    virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> initiate(const Eigen::VectorXd& measurement) const;
    // 预测步骤：基于运动模型推进状态并增加过程噪声
    virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance
    ) const;
    // 将状态空间投影到观测空间，返回投影均值与观测噪声协方差
    virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> project(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance
    ) const;
    // 批量预测：对多条轨迹的状态矩阵和协方差列表同时执行预测
    virtual std::pair<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> multiPredict(
        const Eigen::MatrixXd& mean,
        const std::vector<Eigen::MatrixXd>& covariance
    ) const;
    // 更新步骤：融合新观测，返回更新后的均值与协方差
    virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance,
        const Eigen::VectorXd& measurement
    ) const;
    // 门控距离：计算多个观测与当前预测状态的距离，用于筛选候选匹配
    // onlyPosition=true 时仅使用位置分量；metric 支持 "maha"(马氏) 和 "gaussian"
    Eigen::VectorXd gatingDistance(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance,
        const Eigen::MatrixXd& measurements,
        bool onlyPosition = false,
        const std::string& metric = "maha"
    ) const;

protected:
    Eigen::MatrixXd motionMat;
    Eigen::MatrixXd updateMat;
    double stdWeightPosition;
    double stdWeightVelocity;
};

class KalmanFilterXYWH : public KalmanFilterXYAH {
public:
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> initiate(const Eigen::VectorXd& measurement) const override;
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance
    ) const override;
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> project(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance
    ) const override;
    std::pair<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> multiPredict(
        const Eigen::MatrixXd& mean,
        const std::vector<Eigen::MatrixXd>& covariance
    ) const override;
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance,
        const Eigen::VectorXd& measurement
    ) const override;
};

} // namespace trackerCpp

#endif // TRACKER_CPP_KALMAN_FILTER_H
