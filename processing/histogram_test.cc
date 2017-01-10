#include <stdio.h>
#include "histogram.h"
#include "histogram_test.h"
#include<vector>
#include<armadillo>
using namespace std;
using namespace arma;

Histogram* process_driver_hist(vector<arma::mat > &d_train,
				      vector<arma::mat> &d_test,
				      arma::colvec &l_train,
				      arma::colvec &l_test){

  Histogram *p;
  p = new Histogram();
  p->set_data_train(d_train);
  p->set_data_test(d_test);
  p->set_labels_train(l_train);
  p->set_labels_test(l_test);
  p->process(); // process is in the NO_PROCESSING_H_ header definition
  p->process();
  //printf("history and/or telegrams are happening");
  return(p);
}
