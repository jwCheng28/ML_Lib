#include "linreg.hpp"
#include <iostream>

namespace Linear {

float LinearRegression::costFunction(){
    Eigen::VectorXd h = X * theta;
    // Mean Square Error
    return (1.0 / (2 * m)) * (h - y).array().pow(2).sum();
}

void LinearRegression::gradientDescent(float alpha, int epochs, std::vector<float>& hist, bool cost){
    for (int i = 0; i < epochs; i++) {
        // th -= (a/m) * (X.dot(th) - y).dot(X)
        theta -= (alpha / m) * (((X * theta) - y).transpose() * X).transpose();
        hist.push_back(costFunction());
        if (cost)
            std::cout << "Epoch: " << i+1 << ", Cost: " << hist[i] << std::endl;
    }
}

void LinearRegression::normalEquation(){
    Eigen::MatrixXd Xt = X.transpose();
    Eigen::MatrixXd I = (Xt * X).inverse();
    theta = (I * Xt) * y;
}

} // Linear Namespace
