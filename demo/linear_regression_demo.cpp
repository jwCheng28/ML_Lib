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

void GDtest(MatrixXd data, int rows, int cols) {
    data = Tensor::normalize(data);
    MatrixXd dataMat = Tensor::addOne(data);
    vector<MatrixXd> train_test_data = Tensor::split(dataMat, 1, rows*0.7);
    vector<MatrixXd> trainXY = Tensor::split(train_test_data[0], 0, cols);
    vector<MatrixXd> testXY = Tensor::split(train_test_data[1], 0, cols);

    Linear::LinearRegression linreg(trainXY[0].rows(), trainXY[0].cols());
    cout << "Linear Regression - Gradient Descent\n";
    vector<float> costHistory;
    linreg.gradientDescent(
            trainXY[0], trainXY[1], 
            1.3, 16, 
            costHistory, true);
    cout << "Result Thetas: \n";
    cout << linreg.getTheta() << "\n\n";
    cout << "Test Accuracy: " << Tensor::similarity(testXY[0] * linreg.getTheta(), testXY[1]) * 100 << "%\n";
}

void NEtest(MatrixXd data, int rows, int cols) {
    MatrixXd dataMat = Tensor::addOne(data);
    vector<MatrixXd> train_test_data = Tensor::split(dataMat, 1, rows*0.7);
    vector<MatrixXd> trainXY = Tensor::split(train_test_data[0], 0, cols);
    vector<MatrixXd> testXY = Tensor::split(train_test_data[1], 0, cols);

    Linear::LinearRegression linreg(trainXY[0].rows(), trainXY[0].cols());
    cout << "Linear Regression - Normal Equation\n";
    linreg.normalEquation(trainXY[0], trainXY[1]);
    cout << "Thetas: \n";
    cout << linreg.getTheta() << "\n\n";
    cout << "Test Accuracy: " << Tensor::similarity(testXY[0] * linreg.getTheta(), testXY[1]) * 100 << "%\n";
}

void test1() {
    CSV_READER *csv_file = new CSV_READER("data/test1.csv");
    cout << "Test Data Loaded\n\n";
    cout << "Data Equation y = 2x\n";
    cout << "Theoretical Theta should be t1=0, t2=2\n\n";

    vector<vector<string>> data = csv_file->loadAll(true);
    int rows = data.size(), cols = data[0].size();
    MatrixXd dataM = csv_file->csvToMat(true, data);
    NEtest(dataM, rows, cols);
    delete csv_file;
}

// TO DO
void test2() {
    // CSV_READER *csv_file = new CSV_READER("data/insurance.csv");
//    unordered_map<int, unordered_map<string, string>> encoding = {
//            {1, {{"female", "0"}, {"male", "1"}}},
//            {4, {{"no", "0"}, {"yes", "1"}}},
//    };
    // encode_data(data, encoding, 1);
    return;
}

int main() {
    test1();
    return 0;
}
