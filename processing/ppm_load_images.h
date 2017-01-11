#ifndef PPM_LOAD_IMAGES_H_
#define PPM_LOAD_IMAGES_H_

#include <armadillo>
#include <vector>
#include <string>

std::vector<arma::mat> ppm_load_images(std::string directory, std::string filename);
string[] fileNames(const char*, int);
int fileCount(const char*)

#endif  