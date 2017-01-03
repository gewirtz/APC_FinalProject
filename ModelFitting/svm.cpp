#include "svm.h"
#include "Optimizer.h"
#include <cfloat>
#include <cmath> 


// initializes a model to choose \beta so as to fit Y = X\beta + \epsilon so as to minimize
// ||Y - X\beta ||_2^2

using namespace std;
using namespace arma;