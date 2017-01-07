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
  ~LinearRegression();
  arma::vec predict(std::vector<arma::mat> input); 
  arma::vec get_exactParams(); //gives exact solution 
  arma::vec gradient(int k);
  arma::vec* get_Params();
  int get_params_size();

 private:
 	arma::mat concatenate(std::vector<arma::mat> input);
 	//trains the model (ie updated \beta) for given data x,y 
  void fit(); 
  	//arma::vec fit_value(arma::vec xi);

 	Optimizer* optim;
  arma::mat x; //regressors
  arma::vec y; //labels
  arma::vec* params; 
  std::set<int> label_set; //possible values y_i can take on
  int num_rows;
  int num_cols; 
  int params_size; 
 };

#endif  // LINEARREGRESSION_H_
