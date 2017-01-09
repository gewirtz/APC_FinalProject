#ifndef HISTOGRAM_TEST_H_
#define HISTOGRAM_TEST_H_

#include "histogram.h"
#include <armadillo>
#include <vector>

Histogram* process_driver_hist(std::vector<arma::mat> &imported_train,
				      std::vector<arma::mat> &imported_test,
				      arma::colvec &imported_labels_train,
				      arma::colvec &imported_labels_test);

#endif
