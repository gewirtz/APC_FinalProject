#include "CrossValidation.h"
#include <math.h>
#include "model.h"
#include <armadillo>

using namespace std;
using namespace arma;

CrossValidation::CrossValidation(double range_start, double range_end, double delta=.01){
	this->param_range_start = range_start;
	this->param_range_end = range_end;
	this->delta = delta;
	//to add : some sort of error check making sure the number of params to test isn't too big
}

CrossValidation::~CrossValidation(){}

void CrossValidation::fitParams(Model *m){
	//to be done
}
