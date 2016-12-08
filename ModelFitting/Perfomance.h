#ifndef PERFORMANCE_H_
#define PERFORMANCE_H_

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <armadillo>

using namespace std;
using namespace arma;


class Performance:  {
 public:

 vec mse(double* label, double* predict_label, int num_datapts);


 };

#endif  // PERFORMANCE_H_
