#ifndef _LINEAR_M_H
#define _LINEAR_M_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>

namespace Linear {

class LinearModel{
    protected:
        Eigen::VectorXd theta;
        int m;
    public:
        LinearModel(int dimension, int features) 
        {
            m = dimension;
            theta = Eigen::VectorXd::Random(features).cwiseAbs();
        }
        void setTheta(Eigen::VectorXd theta) 
        {
            this -> theta = theta;
        }
        Eigen::VectorXd getTheta() 
        {
            return theta;
        }
        virtual float costFunction(Eigen::MatrixXd X, Eigen::VectorXd y) = 0;
        virtual void gradientDescent(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, float alpha, int epochs, std::vector<float>& hist, bool cost) = 0;
};

} // Linear Namespace

#endif
