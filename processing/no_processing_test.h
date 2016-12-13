#ifndef NO_PROCESSING_TEST_H_
#define NO_PROCESSING_TEST_H_

#include "no_processing.h"
#include <armadillo>
#include <vector>

No_processing process_driver(std::vector<arma::mat > *imported_train, 
                             std::vector<arma::mat> *imported_test, 
                             arma::colvec *imported_labels_train, 
			     arma::colvec *imported_labels_test);

#endif
