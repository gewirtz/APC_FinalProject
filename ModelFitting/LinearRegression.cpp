#include "LinearRegression.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 

// initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2

using namespace std;
using namespace arma;

  LinearRegression::LinearRegression(vector<arma::mat> train, arma::colvec labels, Optimizer *optim){
  
    this->x = concatenate(train);  //rows contain the ith example, columns contain all instances of a feature
  	this->y = labels; //y_i = label of ith training example
  	this->optim = optim;
  	this->params = zeros<vec>(num_rows*num_cols); //initialize beta in above formulation
  	fit();  //fit beta
    for(int i = 0; i < y.size(); i++){
       this->label_set.insert(y(i));
    }
  } 
  
  mat LinearRegression::concatenate(vector<arma::mat> input){
    if(input == NULL){
      cerr << "Null input\n" << endl;
      exit(-1);
    }
    int num_examples = input.size();
    if(num_examples <= 0){
      cerr << "Need an input\n" << endl;
      exit(-1);
    }
    int num_rows = input[0].n_rows();
    int num_cols = input[0].n_cols() + 1; //regress on pixels and a constant
    mat data = mat(num_examples,num_rows * num_cols);
    for(int i=0;i<num_examples;i++){
      if(input[i].n_rows()!=num_rows || input[i].n_cols()!=num_cols - 1){
        cerr << "Need all input data to have same dimensions\n" << endl;
        exit(-1);
      }
      for(int j=0;j<num_rows;j++){
        for(int k=0;k<num_cols;k++){
          if(k == 0){
            data(i,j*num_cols+k)=1.0; //constant for regression
          }
          else{
            data(i,j*num_cols+k)=input[i](j,k);
          }
        }
      }
    }
    return(data);
  }

  vec LinearRegression::predict(vector<arma::mat> input){
    mat test = concatenate(input);
    if(input.n_cols() != x.n_cols() - 1){
      err << "Need test to have same number of features as training data\n" << endl;
      exit(-1);
    }
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
  
  vec LinearRegression::get_exactParams(){
   	return(pinv(x.t() * x) * x.t() * y);
  }

  vec LinearRegression::get_Params(){
  	return(params);
  }
  
  void LinearRegression::fit(){
  	optim->fitParams(this);
  }
 	
  vec LinearRegression::gradient(){
  	vec grad = zeros<vec>(x.n_cols());
    double fitted_val;
  	for(int i = 0; i < x.n_cols(); i++){
  		for(int j = 0; j < x.n_rows();j++){
        fitted_val = conv_to<double>::from(x.row(j)*params);
	  		grad(i) += -(y(j) - fitted_val)*x(j,i);
	  	}
  	}
  	return(grad); 
  }


