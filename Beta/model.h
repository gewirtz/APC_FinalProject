#ifndef MODEL_H_
#define MODEL_H_

#include <armadillo>

class Optimizer;

class Model {
 public:
  virtual arma::mat predict();
  virtual arma::vec gradient();

  arma::vec params; 

 private:

 	virtual void fit(); 	
 	Optimizer* optim;
};

#endif  // MODEL_H_
