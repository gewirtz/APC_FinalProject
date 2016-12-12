#ifndef DATA_PROCESS_BASE_H_
#define DATA_PROCESS_BASE_H_

#include <armadillo>

class data_processor {
 public:
  virtual data_processor()=0;
  virtual ~data_processor()=0;
  virtual arma::mat process()=0;
 private:
  arma::vec labels;
  arma::mat data;
};

#endif  // DATA_PROCESSOR_H
