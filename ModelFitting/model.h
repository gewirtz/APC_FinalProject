#ifndef MODEL_H_
#define MODEL_H_

#include <armadillo>
#include <vector>

class Optimizer;

class Model {
 public:
  virtual ~Model() {}
  virtual arma::vec predict(std::vector<arma::mat> input) = 0;
  virtual arma::vec gradient(int k) = 0;
  virtual int get_params_size() = 0;

  
  arma::vec *params; 
  int params_size; 
};

#endif  // MODEL_H_
