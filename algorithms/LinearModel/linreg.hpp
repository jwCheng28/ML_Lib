#ifndef _LINEAR_REG_H
#define _LINEAR_REG_H

#include "linear.hpp"

namespace Linear {

class LinearRegression : public LinearModel{
    public:
        LinearRegression(Eigen::MatrixXd X, Eigen::VectorXd y) : LinearModel(X, y) {}
        float costFunction();
        void gradientDescent(float alpha, int epochs, std::vector<float>& hist, bool cost);
        void normalEquation();
};

} // Linear Namespace

#endif
