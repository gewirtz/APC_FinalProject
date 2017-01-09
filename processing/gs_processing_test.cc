#include <stdio.h>
#include "gaussian_smoothing.h"
#include "gs_processing_test.h"
#include<vector>
#include<armadillo>
using namespace std;
using namespace arma;

Gaussian_smoothing* process_driver_gs(vector<arma::mat > &d_train,
                              vector<arma::mat> &d_test,
                              arma::colvec &l_train,
                              arma::colvec &l_test){

  Gaussian_smoothing *p;
  p = new Gaussian_smoothing();
  p->set_data_train(d_train);
  p->set_data_test(d_test);
  p->set_labels_train(l_train);
  p->set_labels_test(l_test);
  p->process(); // process is in the NO_PROCESSING_H_ header definition
  p->process();
  printf("something gaussian is happening");
  return(p);
}
