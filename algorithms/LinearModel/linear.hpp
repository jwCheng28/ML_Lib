#ifndef _LINEAR_M_H
#define _LINEAR_M_H

#include <eigen3/Eigen/Dense>
#include <vector>

namespace Linear {

class LinearModel{
    protected:
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        Eigen::VectorXd theta;
        int m;
    public:
        LinearModel(Eigen::MatrixXd X, Eigen::VectorXd y) 
        {
            this -> X = X;
            this -> y = y;
            m = X.rows();
            theta = Eigen::VectorXd::Random(X.cols());
        }
        void setTheta(Eigen::VectorXd theta) 
        {
            this->theta = theta;
        }
        Eigen::VectorXd getTheta() 
        {
            return theta;
        }
        virtual float costFunction() = 0;
        virtual void gradientDescent(float alpha, int epochs, std::vector<float>& hist, bool cost) = 0;
};

} // Linear Namespace

#endif
