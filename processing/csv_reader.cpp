#include "csv_reader.hpp"
#include <iostream>
#include <boost/algorithm/string.hpp>

void CSV_ITER::restart(bool re){
    if (re) {
        datafile.clear();
        datafile.seekg(0);
    }
}

CSV_READER::CSV_READER(std::string filename){
    iter.datafile.open(filename);
    std::string _;
    iter.row = 0;
    while (getline(iter.datafile, _)) {iter.row++;}
    iter.col = getRow(1).size();
    iter.restart(1);
}

CSV_READER::~CSV_READER(){
    iter.datafile.close();
}

std::vector<int> CSV_READER::shape(){
    return {iter.row, iter.col};
}

std::vector<std::string> CSV_READER::getRow(bool start){
    iter.restart(start);
    std::string cur;
    std::vector<std::string> line;
    if (getline(iter.datafile, cur))
        boost::algorithm::split(line, cur, boost::is_any_of(","));
    for (std::string& a: line)
        if (a == std::string("")) {a = std::string("0");}
    return line;
}

std::vector<std::vector<std::string>> CSV_READER::loadAll(bool start){
    iter.restart(start);
    std::vector<std::vector<std::string>> data;
    while (!iter.datafile.eof())
        data.push_back(getRow());
    return data;
}

Eigen::VectorXd CSV_READER::rowToVect(std::vector<std::string> row){
    int len = row.size();
    Eigen::VectorXd vect(len);
    for (int i = 0; i < len; i++)
        vect[i] = std::stof(row[i]);
    return vect;
}

Eigen::MatrixXd CSV_READER::csvToMat(bool header, std::vector<std::vector<std::string>> csv){
    int row = iter.row - header;
    Eigen::MatrixXd mat(row, iter.col);
    if (csv.size()) {
        for (int i = 0; i < row; i++)
            mat.row(i) = rowToVect(csv[i+header]).transpose();
    } else {
        int cur = iter.datafile.tellg();
        if (header) {getRow(1);}
        mat.row(0) = rowToVect(getRow(!header)).transpose();
        for (int i = 1; i < row; i++)
            mat.row(i) = rowToVect(getRow()).transpose();
        iter.datafile.seekg(cur, iter.datafile.beg);
    }
    return mat;
}
