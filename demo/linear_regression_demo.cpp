#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <eigen3/Eigen/Dense>
#include "../processing/csv_reader.hpp"
#include "../processing/tensor.hpp"
#include "../algorithms/LinearModel/linear.hpp"
#include "../algorithms/LinearModel/linreg.hpp"

using namespace std;
using namespace Eigen;

void encode_data(
        vector<vector<string>>& data, 
        unordered_map<int, unordered_map<string, string>> encode,
        bool header) {

    for (int r = header; r < data.size(); ++r) {
        for (int c = 0; c < data[r].size(); ++c) {
            if (encode.find(c) != encode.end()) {
                data[r][c] = encode[c][data[r][c]];
            }
        }
    }
}

void display(vector<vector<string>>& data) {
    for (auto& row : data) {
        for (auto& e : row) cout << e << ' ';
        cout << '\n';
    }
}

int main() {

    // CSV_READER *csv_file = new CSV_READER("data/insurance.csv");
    CSV_READER *csv_file = new CSV_READER("data/test1.csv");
    vector<vector<string>> data = csv_file->loadAll(true);
    int rows = data.size(), cols = data[0].size();

    unordered_map<int, unordered_map<string, string>> encoding = {
            {1, {{"female", "0"}, {"male", "1"}}},
            {4, {{"no", "0"}, {"yes", "1"}}},
    };
    // encode_data(data, encoding, 1);

    MatrixXd dataM = csv_file->csvToMat(true, data);
    MatrixXd dataMat = Tensor::addOne(dataM);
    vector<MatrixXd> train_test_data = Tensor::split(dataMat, 1, rows*0.8);
    vector<MatrixXd> trainXY = Tensor::split(train_test_data[0], 0, cols);
    vector<MatrixXd> testXY = Tensor::split(train_test_data[1], 0, cols);
    
//    cout << "trainX: \n";
//    cout << trainXY[0] << '\n';
//    cout << "trainY: \n";
//    cout << trainXY[1] << '\n';


    cout << "items: " << trainXY[0].rows() << "  | features: " << trainXY[0].cols() << '\n';
    Linear::LinearRegression linreg(trainXY[0].rows(), trainXY[0].cols());

//    vector<float> costHistory;
//    linreg.gradientDescent(trainXY[0], trainXY[1], 0.00012, 10, costHistory, true);
//    cout << "Thetas: \n";
//    cout << linreg.getTheta() << '\n';

    linreg.normalEquation(trainXY[0], trainXY[1]);
    cout << "Cost: " << linreg.costFunction(trainXY[0], trainXY[1]) << '\n';
    cout << "Thetas: \n";
    cout << linreg.getTheta() << '\n';

    cout << "Accuracy: " << Tensor::similarity(testXY[0] * linreg.getTheta(), testXY[1]) * 100 << '\n';

    return 0;
}
