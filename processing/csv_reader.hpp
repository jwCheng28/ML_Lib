#ifndef _CSV_READ_HPP
#define _CSV_READ_HPP

#include <vector>
#include <string>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <boost/any.hpp>

class CSV_ITER {
    public:
        std::ifstream datafile;
        int row;
        int col;
        void restart(bool re);
};

class CSV_READER {
    private:
        CSV_ITER iter;
    public:
        CSV_READER(std::string filename);
        ~CSV_READER();
        std::vector<int> shape();
        std::vector<std::string> getRow(bool start = 0);
        std::vector<std::vector<std::string>> loadAll(bool start = 0);
        std::vector<std::vector<boost::any>> toAllType(std::vector<std::vector<std::string>>& data);
        Eigen::VectorXd rowToVect(std::vector<std::string> row);
        Eigen::MatrixXd csvToMat(bool header, std::vector<std::vector<std::string>> csv = {});
};

#endif
