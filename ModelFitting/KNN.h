#ifndef KNN_H_
#define KNN_H_

#include "model.h"
#include <vector>
#include <set>

class Optimizer;

class KNN : public Model {
 public:
  KNN(arma::mat train, arma::colvec labels, Optimizer *optim, bool normalize = true); 
  ~KNN();
  
  arma::vec predict(arma::mat testset); 
  arma::vec predict_on_subset(arma::mat test, arma::mat train, int k, arma::vec train_labels, arma::mat dists);
  void set_Params(int k, arma::vec); //sets params so params[k] = p
  std::vector<arma::vec> get_Params();
  arma::vec getLabels();
  int get_num_examples();
  arma::mat getTrainset();
  arma::vec predict_on_subset(arma::mat subset, double k);
  std::set<int> getLabelSet();
 
 private:
  arma::mat standardize(arma::mat data);
  arma::vec internal_predict(arma::mat input, arma::mat train, int k, arma::vec train_labels, arma::mat dists);
  void fit();   //trains the model (ie performs cross validation to choose k) for given data x,y 
 	bool trained;
  bool normalize;
  Optimizer* optim;
  int num_examples;
  arma::mat x; //training set
  arma::vec y; //labels
  std::vector<arma::vec> params; //k 
  std::set<int> label_set; //possible values y_i can take on
  arma::rowvec tr_means;
  arma::rowvec tr_stdev;
  std::vector<bool> remove; 
  int num_regressors;
  int initial_regressors;
 };

#endif  // KNN_H_
