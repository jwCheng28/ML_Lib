#ifndef _PROCESS_HPP
#define _PROCESS_HPP

#include <eigen3/Eigen/Dense>
#include <vector>

namespace Tensor {

Eigen::VectorXd mean(const Eigen::MatrixXd mat);
Eigen::VectorXd std(const Eigen::MatrixXd mat);
Eigen::MatrixXd normalize(const Eigen::MatrixXd mat);
float similarity(const Eigen::VectorXd v1, const Eigen::VectorXd v2);
std::vector<Eigen::MatrixXd> split(const Eigen::MatrixXd mat, int axis, int size);
Eigen::MatrixXd addOne(const Eigen::MatrixXd mat);

}

#endif
