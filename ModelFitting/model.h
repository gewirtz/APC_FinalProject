#ifndef MODEL_H_
#define MODEL_H_

#include <armadillo>
#include <vector>

class Optimizer;

class Model {
 public:
  virtual ~Model() {}
  virtual arma::vec predict(std::vector<arma::mat> input) = 0;

  //computes gradient using the examples lower to upper
  virtual std::vector<arma::vec> gradient(int lower,int upper) = 0; 
  virtual void set_Params(int k, arma::vec p) = 0;
  virtual std::vector<arma::vec> get_Params() = 0;
  virtual int get_num_examples() = 0;
  virtual arma::mat getTrainset() = 0;
  virtual arma::vec getLabels() = 0;
  virtual double cost(int lower, int upper, int k) = 0;
  virtual arma::vec predict_on_subset(arma::mat test, arma::mat train, int k, arma::vec train_labels, arma::mat dists) =0;
  
 private:
  std::vector<arma::mat> params; 
  int num_examples;
};

#endif  // MODEL_H_
