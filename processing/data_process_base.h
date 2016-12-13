#ifndef DATA_PROCESS_BASE_H_
#define DATA_PROCESS_BASE_H_

#include <vector>
#include <armadillo>
#include <string.h>

class Data_processor {
 public:
  virtual Data_processor();
  virtual ~Data_processor();
  
  virtual void process()=0;
 
  virtual void set_data_train(std::vector<arma::mat>* d_train)=0;
  virtual void set_data_test(std::vector<arma::mat>* d_test)=0;
  virtual void set_labels_train(arma::colvec* l_train)=0;
  virtual void set_labels_test(arma::colvec* l_test)=0;
 
  virtual arma::colvec* get_labels_train();
  virtual arma::colvec* get_labels_test();
  virtual std::vector<arma::mat >* get_data_train();
  virtual std::vector<arma::mat>*  get_data_test();

 private:
  int has_been_processed; //switch for whether data has been processed
  arma::colvec labels_train;
  arma::colvec labels_test;
  std::vector<arma::mat > data_train; //This will be the processed data for each class
  std::vector<arma::mat > data_test; //This will be the processed data for each class
};

#endif  // DATA_PROCESSOR_H
