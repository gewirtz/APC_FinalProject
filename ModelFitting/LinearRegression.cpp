#include "LinearRegression.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 


// initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2

using namespace std;
using namespace arma;

LinearRegression::LinearRegression(vector<arma::mat> train, arma::colvec labels, Optimizer *optim){
  this->num_examples = train.size();
  if(num_examples <= 0){
    cerr << "Need an input\n" << endl;
    exit(-1);
  }

  this->x = concatenate(train);  //rows contain the ith example, columns contain all instances of a feature
  this->y = labels; //y_i = label of ith training example
  this->optim = optim;

  vector<vec> temp; 
  vec v;
  temp.push_back(v.zeros(x.n_cols));
  this->params = temp;
  fit();  //fit beta  

  for(int i = 0; i < y.size(); i++){
    this->label_set.insert(y(i));
  }
} 


LinearRegression::~LinearRegression() { 
  //delete [] params; 
}


vec LinearRegression::predict(vector<arma::mat> input){
  mat test = concatenate(input);
  vec labels = test * params[0]; //non_integer fit
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


vec LinearRegression::get_exactParams(){ 
  return(pinv(x.t() * x) * x.t() * y);
}

/*vec LinearRegression::gradient(int k){
  if(k < 0 || k >= params.size()){
    cerr << "Index " << k << " out of bounds.  Need in range 0 " << params.size() << endl;
  }
  vec grad;
  grad = grad.zeros(x.n_cols);
  vec predictions = x * params[k]; //Y = X\beta
  vec resid = y - predictions;
  for(int i = 0; i < x.n_rows;  i++){
    for(int j = 0; j < x.n_cols;j++){
      grad(j) += resid(i) * x(i,j);
    }
  }
  return(1.0/x.n_rows*grad);
}*/

vector<vec> LinearRegression::gradient(int k){
  if(k < 0 || k >= x.n_rows){
    cerr << "Index " << k << " out of bounds.  Need in range 0, " << x.n_rows << endl;
  }
  
  vec grad;
  grad = grad.zeros(x.n_cols);

  vec prediction = x.row(k) * params[0]; //Y = X\beta
  double resid = y[k] - prediction[0];  

  for(int j = 0; j < x.n_cols;j++){
      grad(j) = resid * x(i,j);
  }
  vector<vec> v;
  v.push_back(grad);
  return(grad);
}

vector<vec> LinearRegression::gradient(){
  vec grad;
  grad = grad.zeros(x.n_cols);
  vec predictions = x * params[0]; //Y = X\beta
  vec resid = y - predictions;

  for(int i = 0; i < x.n_rows;  i++){
    for(int j = 0; j < x.n_cols;j++){
      grad(j) += resid(i) * x(i,j);
    }
  }
  vector<vec> v;
  v.push_back(1.0/x.n_rows*grad)
  return(v);
}


void LinearRegression::set_Params(int k, arma::vec p){
  if(k < 0 || k >= params.size()){
    cerr << "Index " << k << " out of bounds.  Need in range 0 " << params.size() << endl;
  }
  params.at(k) = p;
}

vector<vec> LinearRegression::get_Params(){
  return(params);
}

mat LinearRegression::getRegressors(){
  return(x);
}

vec LinearRegression::getLabels(){
  return(y);
}

int LinearRegression::get_num_examples(){
  return(num_examples);
}


mat LinearRegression::concatenate(vector<arma::mat> input){
  int num_rows = input[0].n_rows;
  int num_cols = input[0].n_cols;

  mat data = mat(num_examples,num_rows * num_cols + 1); //includes constant column
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

void LinearRegression::fit(){
  optim->fitParams(this, true); //stochastic gradient descent
}


