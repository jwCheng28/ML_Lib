#ifndef _LOGISTIC_REG_H
#define _LOGISTIC_REG_H

#include "linear.hpp"

namespace Linear {

class LogisticRegression : public LinearModel{
    public:
        LogisticRegression(int dimension, int features) : LinearModel(dimension, features) {}
        Eigen::MatrixXd sigmoid(Eigen::MatrixXd mat);
        float costFunction(Eigen::MatrixXd X, Eigen::VectorXd y);
        void gradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, float alpha, int epochs, std::vector<float>& hist, bool cost);
};

} // Linear Namespace

#endif
