#include "LogisticRegression.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 


// initializes a model to choose weights so as to fit the logistic function
// \frac{1}{1+e^{-\sum_k w_kx_k}}

using namespace std;
using namespace arma;

LogisticRegression::LogisticRegression(vector<arma::mat> train, arma::colvec labels, Optimizer *optim){
  this->num_rows = train[0].n_rows;
  this->num_cols = train[0].n_cols;
  this->x = concatenate(train);  //rows contain the ith example, columns contain all instances of a feature
  this->y = labels; //y_i = label of ith training example
  this->optim = optim;
  this->params = this->params.zeros(x.n_cols); //initialize beta in above formulation
  fit();  //fit weights  
  for(int i = 0; i < y.size(); i++){
    this->label_set.insert(y(i));
  }
} 

vec LogisticRegression::predict(vector<arma::mat> input){
  mat test = concatenate(input);
  vec labels = test * params; //non_integer fit
  int closest = 0;
  double distance;
  double temp;
  for(int i=0;i<input.size();i++){ //round it
    distance = DBL_MAX;
    for(int lab : label_set){
      temp = std::abs(labels[i] - lab); //find closest label
      if(temp <= distance){
        distance = temp;
        closest = lab;
      }
    }
    labels[i] = closest;
  }
  return(labels);
}

vec LogisticRegression::get_exactParams(){
  return(pinv(x.t() * x) * x.t() * y);
}

vec LogisticRegression::get_Params(){
  return(params);
}


mat LogisticRegression::concatenate(vector<arma::mat> input){
/*if(input == NULL){
cerr << "Null input\n" << endl;
exit(-1);
}*/
  int num_examples = input.size();
  //cout << "Given " << num_examples << " examples " << endl;
//cout << "Num examples: " << num_examples << endl;
  if(num_examples <= 0){
    cerr << "Need an input\n" << endl;
    exit(-1);
  }
  mat data = mat(num_examples,num_rows * num_cols + 1);
  for(int i=0;i<num_examples;i++){
    if(input[i].n_rows!=num_rows || input[i].n_cols!=num_cols ){
      cerr << "Need all input data to have same dimensions\n" << endl;
      exit(-1);
    }
    data(i,0) = 1.0; //regress on constant
    for(int j=0;j<num_rows;j++){
      for(int k=0;k<num_cols ; k++){
          data(i,j*num_cols+k+1)=input[i](j,k);
      }
    }
  }
  return(data);
}



void LogisticRegression::fit(){
  optim->fitParams(this);
}

vec LogisticRegression::gradient(){
  vec grad;
  grad = grad.zeros(x.n_cols);
  vec predictions = x * params; //Y = X\beta
  vec resid = predictions - y;
  for(int i = 0; i < x.n_rows;   i++){
    for(int j = 0; j < x.n_cols;j++){
      grad(j) += resid(i) * x(i,j);
    }
  }
  return(1.0/x.n_rows*grad);
}
