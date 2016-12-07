#include "GradientDescent.h"
#include "LinearRegression.h"

int main(int argc, char const *argv[]){
	double train = atof(argv[1]); //assumed we are getting this from somewhere and it is normalized
	double labels = atof(argv[2]);
	double *train2 = &train;
	double *labels2 = &labels;
	GradientDescent *gd = new GradientDescent(100000, .001, .0001) ;
	LinearRegression *fit = new LinearRegression(train2, labels2, 0,0, gd);
	return(0);
}