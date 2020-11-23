#include "decision.hpp"
#include <unordered_map>
#include <unordered_set>

namespace Tree {

std::unordered_map<float, float> label(std::vector<std::vector<float>> mat){
    std::unordered_map<float, float> labels;
    for (std::vector<float> r: mat) {
        for (float c: r) {
            if (!labels.count(c))
                labels[c] = 1;
            else
                labels[c]++;
        }
    }
    return labels;
}

class Test{
    private:
        int col;
        float val;
    public:
        Test(int c, float v) : col(c), val(v) {}
        ~Test() {}
        bool score(std::vector<float> row) {return row[c] >= v;}
};

class Leaf{
    private:
        std::unordered_map<float, float> predictions;
    public:
        Leaf(std::vector<std::vector<float>> mat) {predictions = label(mat);}
        ~Leaf() {}
};

class Decision{
    private:
        Test test;
        DecisionTree true_r;
        DecisionTree false_l;
    public:
        Decision(Test t, DecisionTree r, DecisionTree l) : test(t), true_r(r), false_l(l) {}
        ~Decision() {}
}

// Decision Tree Class
std::vector<std::vector<std::vector<float>>> DecisionTree::splitTree(Test test){
    std::vector<std::vector<float>> pass, fail;
    for (std::vector<float> r: mat)
        test.score(r) : pass.push_back(r) ? fail.push_back(r);
    return {pass, fail};
}

float DecisionTree::gini(std::vector<std::vector<float>> data){
    std::unordered_map<float, float> labels = label(data);
    float c = labels.size();
    float impurity = 0;
    for (auto& l: labels)
        impurity += ((l.second / c) * (1 - l.second / c));
    return impurity;
}

float DecisionTree::gain(std::vector<std::vector<float>> left, std::vector<std::vector<float>> right, float entropy){
    float left_size = float(left.size()) / (left.size() + right.size());
    float right_size = 1 - left_size;
    float weighted_child_entropy = left_size * gini(left) + right_size * gini(right);
    return entropy - weighted_child_entropy;
}

std::tuple<float, Test> DecisionTree::best_split(std::vector<std::vector<float>> data){
    int features = data[0].size() - 1, bgain = -1;
    Question bq;
    float entropy = gini(data);
    for (int i = 0; i < features; i++) {
        // Need a faster way to iterate through columns                
    }
}

DecisionTree DecisionTree::constructTree(){

}

} // Tree Namespace
