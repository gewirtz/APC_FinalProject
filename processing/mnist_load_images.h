#ifndef MNIST_LOAD_IMAGES_H_
#define MNIST_LOAD_IMAGES_H_

#include <armadillo>
#include <vector>
#include <string>

std::vector<arma::mat> mnist_load_images(std::string directory, std::string filename);

#endif
