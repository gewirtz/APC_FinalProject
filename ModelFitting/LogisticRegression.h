#ifndef LOGISTICREGRESSION_H_
#define LOGISTICREGRESSION_H_


#include "GradientModel.h"
#include <vector>
#include <set>


// initializes a model to classify via the probability model 
// P(Y=j|X) = \frac{\exp(\theta_j*X_i)}/{1+\sum_k\exp(\theta_k*X_i)} 

class Optimizer;

class LogisticRegression : public GradientModel  {
 public:
  LogisticRegression(arma::mat train, arma::colvec labels, Optimizer *optim, bool normalize = true); 
  ~LogisticRegression();

  arma::vec predict(arma::mat test); 

  //gradient computed using examples lower to upper
  std::vector<arma::vec> gradient(int lower, int upper); 
  
  void set_Params(int k, arma::vec p); //sets params.at(k) = p
  std::vector<arma::vec> get_Params();
  arma::mat getTrainset();
  arma::vec getLabels();
  int get_num_examples();
  double cost(int lower, int upper, int k); //cost of fitting examples lower to upper
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
  std::vector<std::vector<int>> ovr_labels;
  void set_ovr_labels();
  arma::rowvec tr_means;
  arma::rowvec tr_stdev;
  std::vector<bool> remove; 
  int num_regressors;
  int initial_regressors;

 };



#endif  // LOGISTICREGRESSION_H_
