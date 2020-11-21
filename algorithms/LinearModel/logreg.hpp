#ifndef _LOGISTIC_REG_H
#define _LOGISTIC_REG_H

#include "linear.hpp"

namespace Linear {

class LogisticRegression : public LinearModel{
    public:
        LogisticRegression(Eigen::MatrixXd X, Eigen::VectorXd y) : LinearModel(X, y) {}
        Eigen::MatrixXd sigmoid(Eigen::MatrixXd mat);
        float costFunction();
        void gradientDescent(float alpha, int epochs, std::vector<float>& hist, bool cost);
};

} // Linear Namespace

#endif
