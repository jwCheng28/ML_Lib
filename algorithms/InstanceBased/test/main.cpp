#include <iostream>
#include "../knn.hpp"

int main(){
    Instance::KNN knn(2, 1);
    Eigen::MatrixXd X(6, 3);
    X << 3, 2, 4,
         1, 0, 2,
         4, 3, 4,
         2, 3, 3,
         3, 4, 3,
         0, 0, 1;
    Eigen::VectorXd y(6);
    y << 3, 1, 3, 2, 3, 1;
    Eigen::MatrixXd tX(5, 3);
    tX << 3, 3, 3,
          0, 1, 1,
          4, 3, 3,
          2, 2, 2,
          2, 1, 0;
    std::vector<int> expected = {3, 1, 3, 2, 1};
    std::cout << "Expected Result:  ";
    for (const auto& v: expected)
        std::cout << v << " ";
    std::cout << std::endl;

    std::vector<int> res;
    res = knn.predict(X, y, tX);
    std::cout << "Actual Result:  ";
    for (const auto& val: res)
        std::cout << val << " ";
    std::cout << std::endl;
    return 0;
}
