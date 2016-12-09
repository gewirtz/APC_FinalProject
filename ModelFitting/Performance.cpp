#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <armadillo>
#include "Performance.h"

using namespace std;
using namespace arma;

double Performance::mse(vec label, vec predict_label, int num_datapts){

int n = label.n_elem;
int m = predict_label.n_elem;

vec dif=(label-predict_label);

double MSE = dot(dif,dif)/n;
  
  return(MSE);  
}



  


 	


