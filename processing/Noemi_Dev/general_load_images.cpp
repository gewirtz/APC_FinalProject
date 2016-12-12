#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <armadillo>
#include "general_load_images.h"
#include <sys/stat.h>

using namespace std;
using namespace arma;

double Performance::mse(vec label, vec predict_label, int num_datapts){

	int n = label.n_elem;
	int m = predict_label.n_elem;

	vec dif=(label-predict_label);

	double MSE = dot(dif,dif)/n;
	  
	return(MSE);  
}

mat Performance::correl(vec label, vec predict_label){
	mat R = cor(label,predict_label);
	return(R);
}
 
double Performance::accuracy(vec label, vec predict_label){
	int n=label.n_elem;
	int correct=0;
	for (int i=0;i<n;i++){
		if(label(i)==predict_label(i)){
			correct++;
		}
		//cout << correct <<endl;

	}
	double perc=correct/n;
  
  return(perc);
}

bool fileExists(const string& filename){
	struct stat buf;
	if (stat(filename.c_str(), &buf) != -1){
		return true;
	}
	return false;
}



  


 	


