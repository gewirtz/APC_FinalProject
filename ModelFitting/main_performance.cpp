#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <armadillo>
#include "LinearRegression.h"
#include "Performance.h"
 
using namespace std;
using namespace arma;

int main(int argc, char const *argv[]){
	
/*	// Training
	double train = atof(argv[1]); //assumed we are getting this from somewhere and it is normalized
	double train_labels = atof(argv[2]);
	double *train2 = &train;
	double *train_labels2 = &train_labels;

	// Testing
	double test = atof(argv[1]); //assumed we are getting this from somewhere and it is normalized
	double test_labels = atof(argv[2]);
	double *test2 = &test;
	double *test_labels2 = &test_labels;*/

	// READ PARAMETERS OBJECT

	// PREDICT TRAIN

	// Calculate STATS on the TRAIN



	// PREDICT TESTING

	// Calculate STATS TESTING


	// Print text file with stats

	double label1 = [0,0,1,1]; 
	double label2 = [0,0,1,1]; 
	//double *train2 = &label1;
	//double *train_labels2 = &train_labels;
	int size=4;

	vec stat1 = Performance::mse(&label1,&label2,&size);

	//printf ("mse: %f \n",stat1);


/*	GradientDescent *gd = new GradientDescent(100000, .001, .0001) ;
	LinearRegression *fit = new LinearRegression(train2, labels2, 0,0, gd);

	LinearRegression *param= new LinearRegression::predict(fit)*/
	return(0);
}