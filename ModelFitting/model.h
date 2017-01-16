/* Author : Chase Perlen and Ariel Gewirtz */

#ifndef MODEL_H_
#define MODEL_H_

#include <armadillo>
#include <vector>
#include <set>

class Optimizer;

class Model {
 public:
  virtual ~Model() {}
  virtual arma::vec predict(arma::mat input) = 0;
  virtual void set_Params(int k, arma::vec p) = 0;
  virtual std::vector<arma::vec> get_Params() = 0;
  virtual int get_num_examples() = 0;
  virtual arma::mat getTrainset() = 0;
  virtual arma::vec getLabels() = 0;
  virtual std::set<int> getLabelSet() = 0;
};

#endif  // MODEL_H_
