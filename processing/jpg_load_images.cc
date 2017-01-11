#include <stdio.h>
#include <stdlib.h>
#include <armadillo>
#include <sys/stat.h>
//#include "jpg_load_images.h"
#include "CImg.h"


using namespace std;
using namespace arma;
using namespace cimg_library;

// CImg<float> image;
// string filePath = "/home/andreas/APC524/Project/data/faces/training_faces/";
// image.load(filePath.c_str());

arma::mat convertToArma(CImg<unsigned char> img){

    //vector<arma::mat> all_images;

    //for(int i=0;i<img->x*img->y;i++){

    // create vector of doubles
    arma::mat cur_img(img.height(),img.width());

    for(int row = 0; row < img.width(); row++){
      for(int col = 0; col < img.height(); col++){

        // adds pixel array - only does 'red' pixel since greyscale
        cur_img(row,col) = (double) img.data(row,col);

      } // end col
    } // end row

    //puts current image at the end of the list
    //all_images.push_back(cur_img);

    //} // end num_img
    return cur_img;
}

int main(){

CImg<unsigned char> src("image.jpg");
int width = src.width();
int height = src.height();
unsigned char* ptr = src.data(15,36); // get pointer to pixel @ 10,10
unsigned char pixel = *ptr;

vector<arma::mat> all_images;

img = convertToArma(src);

}