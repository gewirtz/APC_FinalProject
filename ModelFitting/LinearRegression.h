#ifndef LINEARREGRESSION_H_
#define LINEARREGRESSION_H_


#include "model.h"
#include <vector>
#include <set>

//initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2

//TODO in future versions - add regularization, add logistic regression, handle y as matrix 

class Optimizer;

class LinearRegression : public Model  {
 public:
  LinearRegression(std::vector<arma::mat> train, arma::colvec labels, Optimizer *optim); 
  arma::vec predict(std::vector<arma::mat> input); 
  arma::vec get_exactParams(); //gives exact solution 
  arma::vec gradient();
  arma::vec get_Params();
  
 private:
 	arma::mat concatenate(std::vector<arma::mat> input);
 	//trains the model (ie updated \beta) for given data x,y 
  	void fit(); 
  	//arma::vec fit_value(arma::vec xi);

 	Optimizer* optim;
   	arma::mat x; //regressors
  	arma::vec y; //labels
  	std::set<int> label_set; //possible values y_i can take on

 };

#endif  // LINEARREGRESSION_H_
