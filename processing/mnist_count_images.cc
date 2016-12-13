#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

int mnist_count_images(string directory, string filename){

  string fname = directory + filename;
  int magic;
  int num_img;
  
  ifstream mnist_image (fname.c_str(), ios::binary);
  assert( mnist_image.is_open() == 1 );

  // read the magic number, but don't need to use it
  mnist_image.read( (char*) &magic, sizeof(magic) );

  // Next is the number of images
  mnist_image.read( (char*) &num_img, sizeof(num_img) );
  num_img = Reverse(num_img);

  return num_img;

}
