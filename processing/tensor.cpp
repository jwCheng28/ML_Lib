#include "tensor.hpp"

namespace Tensor {

Eigen::VectorXd mean(const Eigen::MatrixXd mat){
    return mat.colwise().mean();
}

Eigen::VectorXd std(const Eigen::MatrixXd mat){
    return ((mat.rowwise() - mean(mat).transpose()).array().square().colwise().sum() 
            / (mat.rows() - 1)).sqrt();
}

Eigen::MatrixXd normalize(const Eigen::MatrixXd mat){
    return (mat.rowwise() - mean(mat).transpose()).array().rowwise() / std(mat).transpose().array();
}

std::vector<Eigen::MatrixXd> splitXY(const Eigen::MatrixXd mat, bool yright){
    int cols = mat.cols();
    if (yright)
        return {mat.leftCols(cols - 1), mat.rightCols(1)};
    else
        return {mat.rightCols(cols - 1), mat.leftCols(1)};
}

std::vector<Eigen::MatrixXd> splitData(const Eigen::MatrixXd mat, float tsize){
    int rows = mat.rows();
    int top = rows * tsize;
    return {mat.topRows(top), mat.bottomRows(rows - top)};
}

} // Tensor Namespace
