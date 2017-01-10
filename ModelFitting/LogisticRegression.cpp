#include "LogisticRegression.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 


// initializes a model to classify via the probability model 
// P(Y=j|X) = \frac{\exp(\theta_j*X_i)}/{1+\sum_k\exp(\theta_k*X_i)} 

using namespace std;
using namespace arma;


LogisticRegression::LogisticRegression(vector<arma::mat> train, arma::colvec labels, Optimizer *optim){
  this->num_examples = train.size();
  if(num_examples <= 0){
    cerr << "Need an input\n" << endl;
    exit(-1);
  }
  srand(506); //shuffle the elements
  this->x = shuffle(concatenate(train));  //rows contain the ith example, columns contain all instances of a feature
  srand(506); //preserve same shuffling
  this->y = shuffle(labels); //y_i = label of ith training example
  this->optim = optim;

  for(int i = 0; i < y.size(); i++){
    this->label_set.insert(y(i));
  }

  vector<vec> temp; 
  vec v;
  temp.push_back(v.zeros(x.n_cols));
  this->params = temp;

  fit();  //fit beta  

}  

LogisticRegression::~LogisticRegression() {}


vec LogisticRegression::predict(vector<arma::mat> input){
  mat test = concatenate(input);
  vec labels(test.n_rows);
  int fitted_val = -1;
  vector<double> label_likelihood(label_set.size());
  vec temp;
  double max_prob;

  for(int i = 0; i < test.n_rows; i++){
    max_prob = 0.0;
    for(int k = 0; k < label_set.size();k++){
      temp = x.row(i) * params[k];
      label_likelihood[k] = 1.0/(1.0+exp(-temp[0]));
      if(label_likelihood[k] > max_prob){
        max_prob = label_likelihood[k];
        fitted_val = label_set[k]; //get this to synch with ari's code
      } 
    }
    labels[i] = fitted_val;
  }
  return(labels);
}



/*
//MAP (maximum aposteriori) fit
vec LogisticRegression::predict(vector<arma::mat> input){
  mat test = concatenate(input);
  vec labels(test.n_rows);
  int fitted_val;
  double max_prob;
  double temp;
  double sum;
  vec fits; //coefficient fits for each class k, given by theta_{k}*x_i

  for(int i = 0; i < test.n_rows;i++){
    fits = fits.zeros(label_set.size());
    max_prob = 0.0;
    sum = 0.0;
    for(int k = 0; k < label_set.size(); k++){
      //fits.at(k) = (test.row(i) *  params.at(k));  //compute denominator of logistic function
      vec v = test.row(i) *  params.at(k);
      fits.at(k) = v[0];
      sum += fits.at(k);
    }
    sum = 1.0 + exp(-sum); //logistic function
    for(int k = 0; k < label_set.size() - 1; k++){
      temp = exp(-fits.at(k))/sum;
      if(temp > max_prob){
        max_prob = temp; //choose the greatest
        fitted_val = k; 
      }
    }
    labels[i] = fitted_val; 
  }
  return(labels);
}
*/

vector<vec> LogisticRegression::gradient(int lower, int upper){ //one v rest fit
  if(lower < 0 || lower >= upper || upper > num_examples){
    cerr << "Lower and upper limits " << lower << " and " << upper << " invalid" << endl;
    cerr << "Need val between 0 and " << num_examples << endl;
    exit(1);
  }
  vector<vec> v;
  vec ovr_lab(upper - lower); //one v rest labels
  vec grad; 
  vec probs;

  for(int k = 0; k < label_set.size(); k++){
    //create one v rest label set for label k
    for(int i = lower; i < upper;  i++){
      if(y[i] == k){
        ovr_lab[i] = 1;
      }
      else{
        ovr_lab[i] = 0;
      }
    }

    grad = grad.zeros(x.n_cols); //reset grad
    probs = 1.0/(1.0+exp(-x*params[k])); 
    
    for(int i = 0; i < x.n_cols; i++){ //i=1...nparams
      for(int j = lower; j < upper; j++){ // j=1...examples
        grad(i) += (ovr_lab[j] - probs(j)) * x(j,i);
      }
    }
    v.push_back(-1.0/x.n_rows*grad);
  }
  return(v); //maximizing likelihood is equivalent to minimizing negative likelihood
}

/*
//zeros<vec>(10)
//MAP (maximum aposteriori) fit
vec LogisticRegression::predict(vector<arma::mat> input){
  mat test = concatenate(input);
  vec labels(test.n_rows);
  int fitted_val;
  double max_prob;
  double temp;
  double sum;
  vec fits; //coefficient fits for each class k, give by theta_{k}*x_i

  for(int i = 0; i < test.n_rows;i++){
    fits = fits.zeros(label_set.size());
    max_prob = 0.0;
    sum = 0.0;
    for(int k = 0; k < label_set.size() - 1; k++){
      //fits.at(k) = (test.row(i) *  params.at(k));  //compute denominator of logistic function
      vec v = test.row(i) *  params.at(k);
      fits.at(k) = v[0];
      sum += fits.at(k);
    }
    sum = 1.0 + exp(-sum); //logistic function
    temp = 1.0 / sum; //probability of the Kth class where label set is given by {0,1,...,K}
    fitted_val = label_set.size() - 1;
    for(int k = 0; k < label_set.size() - 1; k++){
      temp = exp(-fits.at(k))/sum;
      if(temp > max_prob){
        max_prob = temp; //choose the greatest
        fitted_val = k; 
      }
    }
    labels[i] = fitted_val; 
  }
  return(labels);
}

*/


void LogisticRegression::set_Params(int k, arma::vec p){
  if(k < 0 || k >= params.size()){
    cerr << "Index " << k << " out of bounds.  Need in range 0 " << params.size() << endl;
  }
  params.at(k) = p;
}


vector<vec> LogisticRegression::get_Params(){
  return(params);
}

mat LogisticRegression::concatenate(vector<arma::mat> input){
  int num_rows = input[0].n_rows;
  int num_cols = input[0].n_cols;
  int ex_count = input.size();
  mat data = mat(ex_count,num_rows * num_cols + 1); //includes constant column
  for(int i=0; i<ex_count; i++){
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



