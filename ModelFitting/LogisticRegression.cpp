#include "LogisticRegression.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 


// initializes a model to classify via the probability model 
// P(Y=j|X) = \frac{\exp(\theta_j*X_i)}/{1+\sum_k\exp(\theta_k*X_i)} 

using namespace std;
using namespace arma;

LogisticRegression::LogisticRegression(vector<arma::mat> train, arma::colvec labels, Optimizer *optim){
  this->num_rows = train[0].n_rows;
  this->num_cols = train[0].n_cols;
  this->x = concatenate(train);  //rows contain the ith example, columns contain all instances of a feature
  this->y = labels; //y_i = label of ith training example
  this->optim = optim;
  for(int i = 0; i < y.size(); i++){
    this->label_set.insert(y(i));
  }
  vector<vec> temp; 
  vec v;
  for(int i = 0; i < label_set.size();i++){
    temp.push_back(v.zeros(x.n_cols)); //one set of parameters per label
  }
  this->params = temp;
  fit();  //fit beta  
} 

LogisticRegression::~LogisticRegression() {}

void LogisticRegression::set_Params(int k, arma::vec p){
  if(k < 0 || k >= params.size()){
    cerr << "Index " << k << " out of bounds.  Need in range 0 " << params.size() << endl;
  }
  params.at(k) = p;
}

//zeros<vec>(10)
//MAP (maximum aposteriori) fit
vec LogisticRegression::predict(vector<arma::mat> input){
  mat test = concatenate(input);
  vec labels(input.n_rows);
  int fitted_val;
  double max_prob;
  double temp;
  double sum;
  vec fits; //coefficient fits for each class k, give by theta_{k}*x_i

  for(int i = 0; i < input.n_rows;i++){
    fits = fits.zeros(label_set.size())
    max_prob = 0.0;
    sum = 0.0;
    for(int k = 0; k < label_set.size() - 1, k++){
      fits.at(k) = params.at(k)*test.row(i);  //compute denominator of logistic function
      sum += fits.at(k);
    }
    sum = 1 + exp(sum); //logistic function
    temp = 1.0 / sum; //probability of the Kth class where label set is given by {0,1,...,K}
    fitted_val = label_set.size() - 1;
    for(int k = 0; k < label_set.size() - 1, k++){
      temp = exp(fits.at(k))/sum;
      if(temp > max_prob){
        max_prob = temp; //choose the greatest
        fitted_val = k; 
      }
    }
    labels[i] = fitted_val; 
  }
  return(labels);
}

vector<vec> LogisticRegression::get_Params(){
  return(params);
}

mat LogisticRegression::concatenate(vector<arma::mat> input){
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

vec LogisticRegression::gradient(int k){ //one v rest fit
  if(k < 0 || k >= params.size()){
    cerr << "Index " << k << " out of bounds.  Need in range 0 " << params.size() << endl;
  }
  vec n_labels(x.n_rows);
  vec n_params = zeros<vec>(x.n_cols);

  for(int i = 0; i < x.n_rows;  i++){
    if(labels[i] == k){
      n_labels[i] = 1;
    }
    else{
      n_labels[i] = 0;
    }
  }

  vec grad;
  grad = grad.zeros(x.n_cols);
  vec probs = exp(x * params);
  for(int i = 0; i < x.n_cols; i++){ //i=1...nparams
    for(int j = 0; j < x.n_rows; j++){ // j=1...examples
      grad(i) += x(j,i)*(n_labels[j] - probs(j) );
    }
  }
  return(-1.0/x.n_rows*grad); //maximize likelihood
}

