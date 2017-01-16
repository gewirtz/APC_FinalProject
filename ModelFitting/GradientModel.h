/* Author : Chase Perlen and Ariel Gewirtz */

#ifndef GRADIENTMODEL_H_
#define GRADIENTMODEL_H_

#include <armadillo>
#include <vector>

#include "model.h"

class GradientDescent;

class GradientModel : public Model {
 public:
  virtual ~GradientModel() {}
  virtual arma::vec predict(arma::mat input) = 0;

  //computes gradient using the examples lower to upper
  virtual std::vector<arma::vec> gradient(int lower,int upper) = 0; 
  virtual void set_Params(int k, arma::vec p) = 0;
  virtual std::vector<arma::vec> get_Params() = 0;
  virtual int get_num_examples() = 0;
  virtual arma::mat getTrainset() = 0;
  virtual arma::vec getLabels() = 0;
  virtual double cost(int lower, int upper, int k) = 0;
  virtual std::set<int> getLabelSet() = 0;
 
 private:
  std::vector<arma::mat> params; 
  int num_examples;
  std::set<int> label_set;
  GradientDescent* optim;
};

#endif  // MODEL_H_
