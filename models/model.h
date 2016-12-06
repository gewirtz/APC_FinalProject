#ifndef MODEL_H_
#define MODEL_H_

#include <armadillo>
#include "Optimizer.h"

class Model {
 public:
  virtual ~Model(){}

  virtual arma::mat predict() = 0;
  virtual arma::vec gradient() = 0;

  arma::vec params; 

 private:

 	virtual void fit() = 0; 	
 	Optimizer *optim = 0;
};

#endif  // MODEL_H_
