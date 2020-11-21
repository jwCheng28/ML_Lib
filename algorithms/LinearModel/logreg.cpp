#include "logreg.hpp"
#include <iostream>

namespace Linear {

Eigen::MatrixXd LogisticRegression::sigmoid(Eigen::MatrixXd mat){
    return 1.0 / (1 + (-1 * mat.array()).exp());
}

float LogisticRegression::costFunction(){
    Eigen::VectorXd h = sigmoid(X * theta);
    return (-1.0 / m) * 
        ((y.array()) * h.array().log() + (1 - y.array()) * (1 - h.array()).log()).sum();
}

void LogisticRegression::gradientDescent(float alpha, int epochs, std::vector<float>& hist, bool cost){
    for (int i = 0; i < epochs; i++) {
        // th -= (a/m) * (sig(X.dot(th)) - y).dot(X)
        theta -= (alpha / m) * ((sigmoid(X * theta) - y).transpose() * X).transpose();
        hist.push_back(costFunction());
        if (cost)
            std::cout << "Epoch: " << i+1 << ", Cost: " << hist[i] << std::endl;
    }
}

} // Linear Namespace
