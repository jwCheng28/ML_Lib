#include <cmath>
#include <tuple>
#include <unordered_map>
#include "knn.hpp"

namespace Instance {

void setFeature(int k, int p) {
    k_neighbor = k;
    minkowski_p = p;
}

float KNN::minkowski(Eigen::VectorXd x1, Eigen::VectorXd x2){
    /*
    output:
        float -> returns minkowski distance of the 2 vector
     */
    if (x1.size() != x2.size())
        throw std::invalid_argument("Vector Length Does Not Match");
    return (x1 - x2).array().abs().pow(minkowski_p).sum();
}

std::vector<int> KNN::predict(Eigen::MatrixXd& train_X, Eigen::VectorXd& train_y, Eigen::MatrixXd& test){
    /*
    param:
        Eigen::MatrixXd& train_X -> training data X (features)
        Eigen::VectorXd& train_y -> training data y (labels)
        Eigen::MatrixXd& test -> testing data X (features)
    output:
        std::vector<int> result -> vector of test data prediction
    */
    std::vector<int> result;
    for (int r = 0; r < test.rows(); r++) {
        // Store Minkowski Distance and Row Index
        std::vector<std::tuple<float, int>> pts_dist;
        for (int i = 0; i < train_X.rows(); i++)
            pts_dist.push_back(std::make_tuple(minkowski(test.row(r), train_X.row(i)), i));
        
        // Get K Nearest Neighbors
        std::sort(pts_dist.begin(), pts_dist.end());
        std::vector<std::tuple<float, int>> _knn(pts_dist.begin(), pts_dist.begin() + k_neighbor);
        
        // Decide Best Label based on Nearest Neighbors (Highest Freq)
        std::unordered_map<int, int> label_count;
        int max_count = INT_MIN, best_label, index;
        float mks_dist;
        for (const auto& nn: _knn) {
            std::tie(mks_dist, index) = nn;
            int label = train_y[index];
            if (++label_count[label] > max_count) {
                max_count = label_count[label];
                best_label = label;
            }
        }
        result.push_back(best_label);
    }
    return result;
}

} // Instance Namespace
