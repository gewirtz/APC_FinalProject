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

//g++ main_performance.cpp Performance.cpp -O2 -larmadillo -llapack -lblas -lgfortran -o test

int main(int argc, char const *argv[]){
	// READ PARAMETERS OBJECT

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

	vec a(10); a.fill(1);
	vec b(10); b.fill(0);

	for (int i=0;i<10;i++){
		cout << a(i)-b(i) <<endl;
	}

	// PREDICT TRAIN

	// Calculate STATS on the TRAIN

	Performance Pf;
	vec stat1, stat2, stat3;
	stat1 = Pf.mse(a,b,10);
	stat2 = Pf.correl(a,b);
	stat3 = Pf.accuracy(a,b);

	cout << stat1 <<endl;
	cout << stat2(0) <<endl;
	cout << stat3 <<endl;

	// PREDICT TESTING

	// Calculate STATS TESTING

	// Print text file with stats

	//printf ("mse: %f \n",stat1);

	return(0);
}