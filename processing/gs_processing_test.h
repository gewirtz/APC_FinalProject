#ifndef GAUSSIAN_SMOOTHING_TEST_H_
#define GAUSSIAN_SMOOTHING_TEST_H_

#include "gaussian_smoothing.h"
#include <armadillo>
#include <vector>

Gaussian_smoothing* process_driver_gs(std::vector<arma::mat> &imported_train,
			      std::vector<arma::mat> &imported_test,
			      arma::colvec &imported_labels_train,
			      arma::colvec &imported_labels_test);

#endif
