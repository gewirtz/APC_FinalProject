#include <stdio.h>
#include "no_processing.h"
#include "no_processing_test.h"
#include<vector>
#include<armadillo>
using namespace std;
using namespace arma;

No_processing* process_driver(vector<arma::mat > &d_train, 
                              vector<arma::mat> &d_test, 
                              arma::colvec &l_train, 
                              arma::colvec &l_test){

  No_processing *p;
  p = new No_processing();
  p->set_data_train(d_train);
  p->set_data_test(d_test);
  p->set_labels_train(l_train);
  p->set_labels_test(l_test);
  p->process(); // process is in the NO_PROCESSING_H_ header definition
  p->process();
  
  return(p);
 }
