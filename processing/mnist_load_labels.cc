#include <iostream>
#include <fstream>
#include <armadillo>
#include <cassert>
#include "mnist_load_labels.h"
#include "reverse.h"

using namespace std;
using namespace arma;

arma::colvec mnist_load_labels(string directory, string filename){

  string fname = directory + filename;

  ifstream mnist_label (fname.c_str(), ios::binary);
  assert(mnist_label.is_open() == 1);

  int magic = 0;
  int num_img = 0;

  // read magic number, don't need it, just need to get past it
  mnist_label.read( (char*) &magic, sizeof(magic) );

  // read number of images
  mnist_label.read( (char*) &num_img, sizeof(num_img) );
  num_img = Reverse(num_img);

  arma::colvec labels = arma::zeros<arma::colvec>(num_img);

  // read the labels
  for(int i = 0; i < num_img; i++){
    unsigned char cur_label = 0;
    mnist_label.read( (char*) &cur_label, sizeof(cur_label) );
    labels(i) = (double) cur_label;
  }

  return labels;
}
