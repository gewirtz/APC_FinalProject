#include "GradientDescent.h"
#include "LinearRegression.h"

int main(int argc, char const *argv[]){
	
	// Training
	double train = atof(argv[1]); //assumed we are getting this from somewhere and it is normalized
	double train_labels = atof(argv[2]);
	double *train2 = &train;
	double *train_labels2 = &train_labels;

	// Testing
	double test = atof(argv[1]); //assumed we are getting this from somewhere and it is normalized
	double test_labels = atof(argv[2]);
	double *test2 = &test;
	double *test_labels2 = &test_labels;


	GradientDescent *gd = new GradientDescent(100000, .001, .0001) ;
	LinearRegression *fit = new LinearRegression(train2, labels2, 0,0, gd);

	LinearRegression *param= new LinearRegression::predict(fit)
	return(0);
}