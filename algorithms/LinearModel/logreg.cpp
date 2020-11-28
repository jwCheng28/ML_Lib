#include "logreg.hpp"
#include <iostream>

namespace Linear {

Eigen::MatrixXd LogisticRegression::sigmoid(Eigen::MatrixXd mat){
    return 1.0 / (1 + (-1 * mat.array()).exp());
}

float LogisticRegression::costFunction(Eigen::MatrixXd X, Eigen::VectorXd y){
    Eigen::VectorXd h = sigmoid(X * theta);
    return (-1.0 / m) * 
        ((y.array()) * h.array().log() + (1 - y.array()) * (1 - h.array()).log()).sum();
}

void LogisticRegression::gradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, float alpha, int epochs, std::vector<float>& hist, bool cost){
    for (int i = 0; i < epochs; i++) {
        // th -= (a/m) * (sig(X.dot(th)) - y).dot(X)
        theta -= (alpha / m) * ((sigmoid(X * theta) - y).transpose() * X).transpose();
        hist.push_back(costFunction(X, y));
        if (cost)
            std::cout << "Epoch: " << i+1 << ", Cost: " << hist[i] << std::endl;
    }
}

} // Linear Namespace
