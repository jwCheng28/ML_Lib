#ifndef _DECISION_H
#define _DECISION_H

// CART DECISION TREE
// Currently only takes in numerical features and labels

#include <vector>
#include <tuple>
#include <set>

namespace Tree {

class Test;
class Leaf;
class Decision;

class DecisionTree{
    private:
        std::vector<std::vector<float>> mat;
        std::vector<std::set<float>> unique;
    public:
        DecisionTree(std::vector<std::vector<float>> dt) : mat(dt) {}
        ~DecisionTree() {}
        void _uniqueFeature();
        std::vector<std::vector<std::vector<float>>> splitTree(Test test);
        float gini(std::vector<std::vector<float>> data);
        float gain(std::vector<std::vector<float>> left, std::vector<std::vector<float>> right, float entropy);
        std::tuple<float, Test> best_split(std::vector<std::vector<float>> data);
        DecisionTree constructTree();
};

} // Tree Namespace
