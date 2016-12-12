#ifndef DATA_PROCESS_BASE_H_
#define DATA_PROCESS_BASE_H_

#include <armadillo>
#include <string.h>

class data_processor {
 public:
  virtual data_processor()=0;
  virtual ~data_processor()=0;
  
  virtual arma::mat process()=0;
 
  virtual void set_Data()=0;
  virtual void set_Labels()=0;

  //virtual int num_img() = 0;
 private:
 // Bill's note: I don't know what to do with these private members
  arma::vec labels;
  arma::mat data;
};

#endif  // DATA_PROCESSOR_H
