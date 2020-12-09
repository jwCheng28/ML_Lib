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

float similarity(const Eigen::VectorXd v1, const Eigen::VectorXd v2){
    Eigen::VectorXd temp = v1 - v2;
    int count = 0;
    for (const int& val : temp)
        count += !val;
    return count / temp.rows();
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

Eigen::MatrixXd addOne(const Eigen::MatrixXd mat){
    Eigen::MatrixXd X(mat.rows(), mat.cols() + 1);
    X.leftCols(1) = Eigen::MatrixXd::Constant(mat.rows(), 1, 1.0);
    X.rightCols(mat.cols()) = mat;
    return X;
}

} // Tensor Namespace
