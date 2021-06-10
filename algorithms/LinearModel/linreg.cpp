#include "linreg.hpp"
#include <iostream>

namespace Linear {

float LinearRegression::costFunction(Eigen::MatrixXd X, Eigen::VectorXd y) {
    Eigen::VectorXd h = X * theta;
    // Mean Square Error
    return (1.0 / (2 * m)) * (h - y).array().pow(2).sum();
}

void LinearRegression::gradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, float alpha, int epochs, std::vector<float>& hist, bool cost) {
    for (int i = 0; i < epochs; i++) {
        // th -= (a/m) * (X.dot(th) - y).dot(X)
        theta -= (alpha / m) * (((X * theta) - y).transpose() * X).transpose();
        hist.push_back(costFunction(X, y));
        if (cost) std::cout << "Epoch: " << i+1 << ", Cost: " << hist[i] << std::endl;
    }
}

void LinearRegression::normalEquation(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    Eigen::MatrixXd Xt = X.transpose();
    theta = (((Xt * X).inverse() * Xt) * y).transpose();
}

} // Linear Namespace
