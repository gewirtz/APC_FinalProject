#ifndef SVM_H_
#define SVM_H_


#include "model.h"
#include <vector>
#include <set>

class Optimizer;

class SVM : public Model  {
 public:
  SVM(std::vector<arma::mat> train, arma::colvec labels, Optimizer *optim); 
  arma::vec predict(std::vector<arma::mat> input); 
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
  	int num_rows;
  	int num_cols; 

 };

#endif  // SVM_H_