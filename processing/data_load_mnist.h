#ifndef DATA_LOAD_MNIST_H_
#define DATA_LOAD_MNIST_H_

#include <string.h>
#include <armadillo>
#include "data_process_base.h"

class data_load_mnist :  public data_process_base {

 public:

  data_load_mnist(string directory, string lbl_fname, string img_fname);
  ~data_load_mnist();

  //int num_img(string directory, string img_fname);
  arma::mat process(string directory, string img_fname);
  arma::colvec labels(string directory, string lbl_fname, num_img);
  
  private:
  
  string directory_;
  string lbl_fname_;
  string img_fname_;
  //int num_img_;

};

#endif //DATA_LOAD_MNIST_H_

