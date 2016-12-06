#include <GradientDescent.h>

using namespace std;
using namespace arma;

GradientDescent::GradientDescent(int iterations, double alpha, double tol){
	this.iterations = iterations;
	this.alpha = alpha;
	this.tol = tol;
}

void GradientDescent::fitParams(Model *m){
	double update;
	for(int i =0; i < iterations; i++){
		update = m->gradient();
		if(update < tol){
			return;
		}
		m->params -= alpha*update;
	}
	cerr << "Did not converge in given number of iterations" << endl;
}



#endif  // OPTIMIZER_H_
