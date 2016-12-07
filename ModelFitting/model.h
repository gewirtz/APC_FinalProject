#ifndef MODEL_H_
#define MODEL_H_

#include <armadillo>

class Optimizer;

class Model {
 public:
  virtual arma::mat predict(double *input, int rows, int cols) = 0;
  virtual arma::vec gradient() = 0;

  arma::vec params; 

};

#endif  // MODEL_H_
