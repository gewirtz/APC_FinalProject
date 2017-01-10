#ifndef KNN_H_
#define KNN_H_

#include "model.h"
#include <vector>
#include <set>

class Optimizer;

class LinearRegression : public Model {
 public:
  KNN(std::vector<arma::mat> train, arma::colvec labels, Optimizer *optim); 
  ~KNN();
  
  arma::vec predict(std::vector<arma::mat> input); 
  arma::vec predict_on_subset(arma::mat test, arma::mat train, int k);
  void set_Params(int k, arma::vec p); //sets params so params[k] = p
  void set_k(int k);
  std::vector<arma::vec> get_Params();
  arma::vec getLabels();
  int get_num_examples();
  arma::mat getTrainset();
  arma::vec predict_on_subset(arma::mat subset, double k);

  // unclear what these are doing for KNN
  std::vector<arma::vec> gradient() = 0;
  std::vector<arma::vec> gradient(int k) = 0;

 private:
 	arma::mat concatenate(std::vector<arma::mat> input);
  arma::vec internal_predict(arma::mat input, arma::mat train, int k, arma::vec train_labels);
  //arma::vec choose_k(arma::mat input); //performs cross validation to choose the best k
  void fit();   //trains the model (ie performs cross validation to choose k) for given data x,y 
 	Optimizer* optim;
  int num_examples;
  int k;
  arma::mat x; //regressors
  arma::vec y; //labels
  std::vec<arma::vec> params; //the average pixel value for each location across all images for each label 
  std::set<int> label_set; //possible values y_i can take on
 };

#endif  // KNN_H_
