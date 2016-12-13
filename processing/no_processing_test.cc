#include <stdio.h>
#include "no_processing.h"
#include<vector>
#include<armadillo>
using namespace std;

No_processing process_driver(vector<arma::mat > *imported_train, 
                              vector<arma::mat> *imported_test, 
                              arma::colvec *imported_labels_train, 
                              arma::colvec *imported_labels_test){

  No_processing p;
  //p=new No_processing();
  p.set_data_train(imported_train);
  p.set_data_test(imported_test);
  p.set_labels_train(imported_labels_train);
  p.set_labels_test(imported_labels_test);
  p.process(); // process is in the NO_PROCESSING_H_ header definition
  printf("something is happening");
  return(p);
 };
