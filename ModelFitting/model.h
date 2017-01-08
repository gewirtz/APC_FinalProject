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
  virtual std::vector<arma::vec> get_Params() = 0;
  virtual void set_Params(int k, arma::vec p) = 0;
  //virtual int get_params_size() = 0;

 private:
  std::vector<arma::vec> params; 
};

#endif  // MODEL_H_
