#ifndef _LINEAR_REG_H
#define _LINEAR_REG_H

#include "linear.hpp"

namespace Linear {

class LinearRegression : public LinearModel{
    public:
        LinearRegression(int dimension, int features) : LinearModel(dimension, features) {}
        float costFunction(Eigen::MatrixXd X, Eigen::VectorXd y);
        void gradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, float alpha, int epochs, std::vector<float>& hist, bool cost);
        void normalEquation(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

} // Linear Namespace

#endif
