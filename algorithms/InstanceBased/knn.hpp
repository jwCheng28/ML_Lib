#ifndef _KNN_H
#define _KNN_H

#include <eigen3/Eigen/Dense>
#include <vector>

namespace Instance {

class KNN{
    /*
    Constructor param:
        int -> k neighbors
        int -> minkowski distance power: 1 - Manhanttan dist, 2 - Euclidean
    */
    private:
        int k_neighbor;
        int minkowski_p;
    public:
        KNN(int k, int p) : k_neighbor(k), minkowski_p(p) {}
        ~KNN() {}
        float minkowski(Eigen::VectorXd x1, Eigen::VectorXd x2);
        std::vector<int> predict(Eigen::MatrixXd& train_X, Eigen::VectorXd& train_y, Eigen::MatrixXd& test);
};

} // Instance Namespace

#endif
