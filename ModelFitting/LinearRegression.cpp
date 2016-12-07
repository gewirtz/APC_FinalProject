#include "LinearRegression.h"
#include "Optimizer.h"

// initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2
// TODO in future versions - make compatible with eigen library 

using namespace arma;

  LinearRegression::LinearRegression(double* train, double* labels, int num_features, int num_datapts, Optimizer *optim){
  	this->x = mat(train, num_datapts, num_features);  //rows contain the ith example, columns contain all instances of a feature
  	this->y = vec(labels,num_datapts); //extend to matrix case? 
  	this->optim = optim;
  	this->params = zeros<vec>(num_features);
  	fit();
  } 
  
  mat LinearRegression::predict(double *input, int rows, int cols){
  	if(cols != y.size()){
  		std::cerr << "Input of incompatible dimension" << endl;
  		exit(-1);
  	}
  	mat data = mat(input,rows,cols);
  	return(data*params);
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
  	vec grad = vec(y.size());
    double fitted_val;
  	for(int i = 0; i < y.size(); i++){
  		grad(i) = 0.0;
  		for(int j = 0; j < x.n_rows;j++){
        fitted_val = conv_to<double>::from(x.row(j)*params);
	  		grad(i) += -(y(j) - fitted_val)*x(j,i);
	  	}
  	}
  	return(grad); 
  }


