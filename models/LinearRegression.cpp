#include "LinearRegression.h"


// initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2
// TODO in future versions - make compatible with eigen library 

using namespace arma;
using namespace std; 


  LinearRegression::LinearRegression(double* train, double* labels, int num_features, int num_datapts, Optimizer *optim){
  	this.x = mat(train, num_datapts, num_features);  //rows contain the ith example, columns contain all instances of a feature
  	this.y = vec(labels,num_datapts); //extend to matrix case? 
  	this.optim = optim;
  	fit();
  } 

  LinearRegression::~LinearRegression() {
  }
  
  mat LinearRegression::predict(double *input, int rows, int cols){
  	if(cols != y.size()){
  		cerr << "Not gonna be able to do it" << endl;
  		exit(-1);
  	}
  	mat input = mat(input,rows,cols);
  	return(mat(input*params));
  }
  
  vec LinearRegression::get_exactParams(){
  	return(pinv(x.t * x) * x.t * y)
  }

  vec LinearRegression::get_params(){
  	return(params);
  }
  
  void LinearRegression::fit(){
  	optim->fit_params(this);
  }
 	
  vec LinearRegression::gradient(){
  	vec grad = vec(y.size());
  	for(int i = 0; i < y.size(); i++){
  		grad[i] = 0.0;
  		for(int j = 0; j < x.n_rows){
	  		grad[i] += -(y[j] - x.row(j)*m->params)*x[j][i];
	  	}
  	}
  	return(grad); 
  }


