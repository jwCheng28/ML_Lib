#ifndef _PROCESS_HPP
#define _PROCESS_HPP

#include <eigen3/Eigen/Dense>
#include <vector>

namespace Tensor {

Eigen::VectorXd mean(const Eigen::MatrixXd mat);
Eigen::VectorXd std(const Eigen::MatrixXd mat);
Eigen::MatrixXd normalize(const Eigen::MatrixXd mat);
std::vector<Eigen::MatrixXd> splitXY(const Eigen::MatrixXd mat, bool yright);
std::vector<Eigen::MatrixXd> splitData(const Eigen::MatrixXd mat, float tsize);

}

#endif
