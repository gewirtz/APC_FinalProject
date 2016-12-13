#ifndef MNIST_LOAD_LABELS_H_
#define MNIST_LOAD_LABELS_H_

#include <armadillo>
#include <string>

arma::colvec mnist_load_labels(std::string directory, std::string filename);

#endif
