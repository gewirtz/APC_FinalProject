#ifndef LINEARREGRESSION_H_
#define LINEARREGRESSION_H_


#include "model.h"
#include <vector>
#include <set>

//initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2...decision as in multinomial probit 

//TODO in future versions - add regularization

class Optimizer;


class LinearRegression : public Model  {
 public:
  LinearRegression(std::vector<arma::mat> train, arma::colvec labels, Optimizer *optim); 
  ~LinearRegression();
  
  arma::vec predict(std::vector<arma::mat> input); 
  arma::vec get_exactParams(); //gives exact solution 
  //gradient computed using examples lower to upper
  std::vector<arma::vec> gradient(int lower, int upper);  //gradient using examples lower (inclusive) to upper (not inclusive)
  
  void set_Params(int k, arma::vec p); //sets params.at(k) = p
  std::vector<arma::vec> get_Params();
  arma::mat getRegressors();
  arma::vec getLabels();
  int get_num_examples();
  double cost(int lower, int upper, int k); //cost of fitting examples lower (inclusive) to upper (not inclusive)

  arma::mat getTrainset();
  void set_k(int k);
  arma::vec predict_on_subset(arma::mat test, arma::mat train, int k, arma::vec train_labels);
  
 private:
 	arma::mat concatenate(std::vector<arma::mat> input);
 
  void fit();   //trains the model (ie updates \beta) for given data x,y 


 	Optimizer* optim;
  int num_examples;
  arma::mat x; //regressors
  arma::vec y; //labels
  std::vector<arma::vec> params; 
  std::set<int> label_set; //possible values y_i can take on
 };

#endif  // LINEARREGRESSION_H_
