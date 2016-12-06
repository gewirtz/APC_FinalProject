#ifndef LINEARREGRESSION_H_
#define LINEARREGRESSION_H_


#include "model.h"

//initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2

//TODO in future versions - add regularization, add logistic regression, handle y as matrix 

class LinearRegression : public Model  {
 public:
  LinearRegression(double* train, double* labels, int num_features, int num_datapts, Optimizer *optim); 

  arma::mat predict(double *input, int rows, int cols);
  arma::vec get_exactFit(); //gives exact solution 
  arma::vec gradient();

  arma::vec params;//fitted coefficients
 private:
 	//trains the model (ie updated \beta) for given data x,y 
  	void fit();

 	Optimizer *optim(int iteratitions, double threshold, double alpha); //allow user input in constructor in future
  	arma::mat x; //regressors
  	arma::vec y; //labels

#endif  // LINEARREGRESSION_H_
