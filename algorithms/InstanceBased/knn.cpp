#include <cmath>
#include <tuple>
#include <unordered_map>
#include "knn.hpp"

namespace Instance {

float KNN::minkowski(Eigen::VectorXd x1, Eigen::VectorXd x2){
    if (x1.size() != x2.size())
        return -1;
    return (x1 - x2).array().abs().pow(minkowski_p).sum();
}

std::vector<int> KNN::predict(Eigen::MatrixXd train_X, Eigen::VectorXd train_y, Eigen::MatrixXd test){
    std::vector<int> result;
    for (int r = 0; r < test.rows(); r++) {
        std::vector<std::tuple<float, int>> pts_dist;
        std::unordered_map<int, int> label_count;
        for (int i = 0; i < train_X.rows(); i++)
            pts_dist.push_back(std::make_tuple(minkowski(test.row(r), train_X.row(i)), i));
        std::sort(pts_dist.begin(), pts_dist.end());
        std::vector<std::tuple<float, int>> _knn(pts_dist.begin(), pts_dist.begin() + k_neighbor);
        int max_c = -1, max_l;
        for (const auto& val: _knn) {
            float v;
            int index;
            std::tie(v, index) = val;
            int label = train_y[index];
            if (!label_count.count(label))
                label_count[label] = 0;
            label_count[label]++;
            if (label_count[label] > max_c) {
                max_c = label_count[label];
                max_l = label;
            }
        }
        result.push_back(max_l);
    }
    return result;
}

} // Instance Namespace
