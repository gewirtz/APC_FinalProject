#ifndef LINEARREGRESSION_H_
#define LINEARREGRESSION_H_


#include "model.h"

//initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2

//TODO in future versions - add regularization, add logistic regression, handle y as matrix 

class Optimizer;

class LinearRegression : public Model  {
 public:
  LinearRegression(double* train, double* labels, int num_features, int num_datapts, Optimizer *optim); 

  arma::mat predict(double *input, int rows, int cols);
  arma::vec get_exactParams(); //gives exact solution 
  arma::vec gradient();
  arma::vec get_Params();

 private:
 	//trains the model (ie updated \beta) for given data x,y 
  	void fit(); 	
 	Optimizer* optim;

   	arma::mat x; //regressors
  	arma::vec y; //labels

 };

#endif  // LINEARREGRESSION_H_
