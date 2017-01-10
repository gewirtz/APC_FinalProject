#ifndef LOGISTICREGRESSION_H_
#define LOGISTICREGRESSION_H_


#include "model.h"
#include <vector>
#include <set>


// initializes a model to classify via the probability model 
// P(Y=j|X) = \frac{\exp(\theta_j*X_i)}/{1+\sum_k\exp(\theta_k*X_i)} 

class Optimizer;

class LogisticRegression : public Model  {
 public:
  LogisticRegression(std::vector<arma::mat> train, arma::colvec labels, Optimizer *optim); 
  ~LogisticRegression();
  
  arma::vec predict(std::vector<arma::mat> input); 
  std::vector<arma::vec> gradient(int k); //gradient for example k, used for stoch gradient descent
  std::vector<arma::vec> gradient(); //gradient for all examples, used for batch gradient descent
  void set_Params(int k, arma::vec p); //sets params.at(k) = p
  std::vector<arma::vec> get_Params();
  arma::mat getRegressors();
  arma::vec getLabels();
  int get_num_examples();

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

#endif  // LOGISTICREGRESSION_H_
