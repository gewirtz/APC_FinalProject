/* Author : Chase Perlen */

//initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2...decision as in multinomial probit 
// regularized model minimizes ||Y - X\beta ||_2^2 + \sum_{i=1}^nrows l2*|| X_i||_2^2



#ifndef LINEARREGRESSION_H_
#define LINEARREGRESSION_H_


#include "GradientModel.h"
#include <vector>
#include <set>


class Optimizer;

class LinearRegression : public GradientModel  {
 public:
  
  LinearRegression(arma::mat train, arma::colvec labels, Optimizer *optim, double l2 = 0.0, bool normalize = true); 
  ~LinearRegression();
  
  arma::vec predict(arma::mat input); 
  arma::vec get_exactParams(); //gives exact solution 
  //gradient computed using examples lower to upper
  std::vector<arma::vec> gradient(int lower, int upper);  //gradient using examples lower (inclusive) to upper (not inclusive)
  
  void set_Params(int k, arma::vec p); //sets params.at(k) = p
  std::vector<arma::vec> get_Params();
  arma::vec getLabels();
  int get_num_examples();
  double cost(int lower, int upper, int k); //cost of fitting examples lower (inclusive) to upper (not inclusive)
  arma::mat getTrainset();
  std::set<int> getLabelSet();

 private:
  arma::mat standardize(arma::mat data);
  void fit();   //trains the model (ie updates \beta) for given data x,y 
  bool trained;
  bool normalize;
 	Optimizer* optim;
  int num_examples;
  arma::mat x; //regressors
  arma::vec y; //labels
  std::vector<arma::vec> params; 
  std::set<int> label_set; //possible values y_i can take on
  arma::rowvec tr_means;
  arma::rowvec tr_stdev;
  std::vector<bool> remove; 
  int num_regressors;
  int initial_regressors;
  double l2; //if != 0, we have regularized regression

 };

#endif  // LINEARREGRESSION_H_
