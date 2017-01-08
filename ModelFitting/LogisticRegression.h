#ifndef LOGISTICREGRESSION_H_
#define LOGISTICREGRESSION_H_


#include "model.h"
#include <vector>
#include <set>


//initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2

//TODO in future versions - add regularization, add logistic regression, handle y as matrix 


class Optimizer;

class LogisticRegression : public Model  {
 public:
  LogisticRegression(std::vector<arma::mat> train, arma::colvec labels, Optimizer *optim); 
  ~LogisticRegression();
  arma::vec predict(std::vector<arma::mat> input); 
  arma::vec gradient(int k);
  void set_Params(int k, arma::vec p); //sets params.at(k) = p
  std::vector<arma::vec> get_Params();
  int get_params_size();

 private:
  arma::mat concatenate(std::vector<arma::mat> input);
  //trains the model (ie updated \beta) for given data x,y 
  void fit(); 
    //arma::vec fit_value(arma::vec xi);

  Optimizer* optim;
  arma::mat x; //regressors
  arma::vec y; //labels
  std::vector<arma::vec> params; 
  std::set<int> label_set; //possible values y_i can take on
  int num_rows;
  int num_cols; 
 };

#endif  // LOGISTICREGRESSION_H_
