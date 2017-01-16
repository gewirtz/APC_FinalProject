#ifndef JPG_LOAD_IMAGES_H_
#define JPG_LOAD_IMAGES_H_

#include <armadillo>
#include <vector>
#include <string>
 
std::vector<arma::mat> jpg_load_images(std::string testing_data);
std::vector<std::string> fileNamesJPG(const char* direc);
arma::mat readJPG(const char *img);
//arma::mat ppm_load_images(std::string directory, std::string filename);
//arma::mat convertToArma(CImg img);
//vector<arma::mat> jpeg_load_images(string testing_data)
//string[] fileNames(const char*, int);
//int fileCount(const char*)

#endif  