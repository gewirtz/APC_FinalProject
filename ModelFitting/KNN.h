#ifndef KNN_H_
#define KNN_H_

#include "model.h"
#include <vector>
#include <set>

class Optimizer;

class KNN : public Model {
 public:
  KNN(std::vector<arma::mat> train, arma::colvec labels, Optimizer *optim); 
  ~KNN();
  
  arma::vec predict(std::vector<arma::mat> input); 
  arma::vec predict_on_subset(arma::mat test, arma::mat train, int k, arma::vec train_labels, arma::mat dists);
  void set_Params(int k, arma::vec); //sets params so params[k] = p
  std::vector<arma::vec> get_Params();
  arma::vec getLabels();
  int get_num_examples();
  arma::mat getTrainset();
  arma::vec predict_on_subset(arma::mat subset, double k);

 private:
 	arma::mat concatenate(std::vector<arma::mat> input);
  arma::vec internal_predict(arma::mat input, arma::mat train, int k, arma::vec train_labels, const arma::mat dists);
  void fit();   //trains the model (ie performs cross validation to choose k) for given data x,y 
 	Optimizer* optim;
  int num_examples;
  arma::mat x; //training set
  arma::vec y; //labels
  std::vector<arma::vec> params; //k 
  std::set<int> label_set; //possible values y_i can take on
 };

#endif  // KNN_H_
