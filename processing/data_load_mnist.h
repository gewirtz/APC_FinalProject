#ifndef DATA_LOAD_MNIST_H_
#define DATA_LOAD_MNIST_H_

#include <string.h>
#include <armadillo>
#include "data_process_base.h"

class data_load_mnist :  public data_process_base {

 public:

  data_load_mnist(string directory, string lbl_fname, string img_fname);
  ~data_load_mnist();

  int num_img(string directory, string img_fname);
  vector<arma::mat> all_images(string directory, string img_fname);
  arma::colvec labels = labels(string directory, string lbl_fname);

};

#endif //DATA_LOAD_MNIST_H_

