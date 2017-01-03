#ifndef MODEL_H_
#define MODEL_H_

#include <armadillo>
#include <vector>

class Optimizer;

class Model {
 public:
  virtual arma::vec predict(std::vector<arma::mat> input) = 0;
  virtual arma::vec gradient() = 0;

  arma::mat params; 

};

#endif  // MODEL_H_
