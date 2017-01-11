#ifndef HYPERMODEL_H_
#define HYPERMODEL_H_

#include <armadillo>
#include <vector>
#include "model.h"
#include <set>

class Optimizer;

class HyperModel : public Model{

 public:
  virtual arma::vec predict(std::vector<arma::mat> input) = 0;
  virtual void set_Params(int k, arma::vec p) = 0;
  virtual std::vector<arma::vec> get_Params() = 0;
  virtual int get_num_examples() = 0;
  virtual arma::mat getTrainset() = 0;
  virtual arma::vec getLabels() = 0;
  virtual arma::vec predict_on_subset(arma::mat test, arma::mat train, int k, arma::vec train_labels, arma::mat dists) =0;
  
 private:
  std::vector<arma::mat> params; 
  int num_examples;
  std::set<int> label_set; //possible values y_i can take on
};

#endif  // HYPERMODEL_H_
