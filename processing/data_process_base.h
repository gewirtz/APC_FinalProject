#ifndef DATA_PROCESS_BASE_H_
#define DATA_PROCESS_BASE_H_

#include <armadillo>
#include <string.h>

class Data_processor {
 public:
  virtual data_processor()=0;
  virtual ~data_processor()=0;
  
  virtual void process()=0;
 
  virtual void set_data_train()=0;
  virtual void set_data_test()=0;
  virtual void set_labels_train()=0;
  virtual void set_labels_test()=0;
 
  virtual arma::colvec* get_labels_train();
  virtual arma::colvec* get_labels_test();
  virtual arma::mat* get_data_train();
  virtual arma::mat* get_data_test()

 private:
  int has_been_processed; //switch for whether data has been processed
  arma::colvec labels_train;
  arma::colvec labels_test;
  arma::mat data_train; //This will be the processed data for each class
  arma::mat data_test; //This will be the processed data for each class
};

#endif  // DATA_PROCESSOR_H
