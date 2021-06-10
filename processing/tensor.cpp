#include "tensor.hpp"
#include <cmath>

namespace Tensor {

Eigen::VectorXd mean(const Eigen::MatrixXd mat) {
    return mat.colwise().mean();
}

Eigen::VectorXd std(const Eigen::MatrixXd mat) {
    return ( ( mat.rowwise() - mean(mat).transpose() ).array().square().colwise().sum() 
            / mat.rows() ).sqrt();
}

Eigen::MatrixXd normalize(const Eigen::MatrixXd mat) {
    Eigen::VectorXd mu = mean(mat);
    Eigen::VectorXd sigma = std(mat);
    return (mat.rowwise() - mu.transpose()).array().rowwise() / sigma.transpose().array();
}

float similarity(const Eigen::VectorXd v1, const Eigen::VectorXd v2) {
    Eigen::VectorXd diff = v1 - v2;
    int similar = 0;
    for (int i = 0; i < diff.size(); ++i) {
        similar += (fabs(diff[i]) < 1e-9);
    }
    return (float) similar / diff.size();
}

std::vector<Eigen::MatrixXd> split(const Eigen::MatrixXd mat, int axis, int size) {
    /*
    param:
        Eigen::MatrixXd mat -> dataset to be split
        int axis -> split by column (0) or rows (1)
        int size -> size to start split
    output:
        std::vector<int> result -> vector of 2 matrix splited from the input matrix
    */

    if (axis == 0) {
        int cols = mat.cols();
        return {mat.leftCols(size), mat.rightCols(cols-size)};
    } else if (axis == 1) {
        int rows = mat.rows();
        return {mat.topRows(size), mat.bottomRows(rows-size)};
    }
}

Eigen::MatrixXd addOne(const Eigen::MatrixXd mat) {
    Eigen::MatrixXd X(mat.rows(), mat.cols() + 1);
    X.leftCols(1) = Eigen::MatrixXd::Constant(mat.rows(), 1, 1.0);
    X.rightCols(mat.cols()) = mat;
    return X;
}

} // Tensor Namespace
