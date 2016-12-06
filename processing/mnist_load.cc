#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <assert.h>

// at the request of the modeling team...
//#include <armadillo>

// heavily inspired by "canonical" mnist data readers

using namespace std;

int Reverse(int i){

  // since data is imported in binary, we have to do
  // to bitwise operations!

  // bitwise operators modify variables considering
  // the bit patterns that represent the values
  // they store

  // & is bitwise addition
  // >> is shift bits to the right

  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;

  int tmp;

  tmp 
    = ((int) ch1 << 24)
    + ((int) ch2 << 16)
    + ((int) ch3 << 8)
    + ch4;

  //cout << test << "\n";

  return tmp; 

}
int load_mnist_img(string directory, string filename,
                    vector<vector<double> > &all_images){

  string fname = directory + filename;
  
  // the example I found imports data in binary:
  ifstream mnist_image (fname.c_str(), ios::binary);
  
  assert( mnist_image.is_open() == 1 );
  
  cout << "processing " << filename << "\n";

  int magic = 0;
  int num_img = 0;
  int rows = 0;
  int cols = 0;

  // The first number in the file is the "Magic Number"

  //The magic number is an integer where the first 2 bytes are zero
  //and the 3rd byte represents the type of code, and the 4th byte 
  //determines the dimensions of the matrix/vector;

  // We need to FLIP the magic number since it is in BIG-ENDIAN
  // as opposed to LITTLE-ENDIAN
  mnist_image.read( (char*) &magic, sizeof(magic) );
  magic = Reverse(magic);

  // Next is the number of images
  mnist_image.read( (char*) &num_img, sizeof(num_img) );
  num_img = Reverse(num_img);

  // rows, cols
  mnist_image.read( (char*) &rows, sizeof(rows) );
  mnist_image.read( (char*) &cols, sizeof(cols) );
  rows = Reverse(rows);
  cols = Reverse(cols);

  // The rest of the data is the image data
  // the individual images are stored in chunks of 32 bits/
  // At this time there is NO RESHAPING

  // opportunity here for parallelization

  for(int i = 0; i < num_img; i++){

    // create vector of doubles
    vector <double> cur_img;

    // all the images are the same size!
    // the images are stored in LITTLE-ENDIAN, no need
    // to reverse --> confusing!
    for(int row = 0; row < rows; row++){
      for(int col = 0; col < cols; col++){

        unsigned char pixel = 0; // 1 byte
        mnist_image.read( (char*) &pixel, sizeof(pixel) );

        // adds pixel to the end of the list
        cur_img.push_back( (double) pixel);

      } // end col
    } // end row

    //puts current image at the end of the list
    all_images.push_back(cur_img); 

  } // end num_img

    return num_img;  

} // end function load_mnist

void load_mnist_label(string directory, string filename, 
                      vector<double> labels){

  string fname = directory + filename;

  ifstream mnist_label (fname.c_str(), ios::binary);
  assert(mnist_label.is_open() == 1);

  int magic = 0;
  int num_img = 0;

  // read magic number, don't need it, just need to get pas it
  mnist_label.read( (char*) &magic, sizeof(magic) );

  // read number of images
  mnist_label.read( (char*) &num_img, sizeof(num_img) );
  num_img = Reverse(num_img);

  // read the labels
  for(int i = 0; i < num_img; i++){
    unsigned char cur_label = 0;
    mnist_label.read( (char*) &cur_label, sizeof(cur_label) );
    labels[i] = (double) cur_label;
  }
}


int main(){

  // these will likely be taken from config file
  string train_directory = "../data/mnist/training/";
  string test_directory = "../data/mnist/testing/";

  string train_lbl = "train-labels.idx1-ubyte";
  string train_img = "train-images.idx3-ubyte";
  string test_lbl = "t10k-labels.idx1-ubyte";
  string test_img = "t10k-images.idx3-ubyte";


  // we are going to put things into an ARMADILLO MATRIX
  // which is really just a vector
  // vector<arma::mat> all_images;

  // DOUBLE VECTOR Implementation

  // load the images
  vector<vector<double> > all_images;
  int num_img;

  num_img = load_mnist_img(train_directory, train_img, all_images);

  // load the labels
  vector<double> labels(num_img);
  load_mnist_label(train_directory, train_img, labels);


  // check to make sure you did it right
  
  cout << all_images.size() << endl; // how many images you have
  cout << all_images[0].size() << endl; // how big each image is
  
  // take a look at an image
  for(int i = 0; i < all_images[0].size(); i++){
    cout << all_images[0][i] << " ";

    if (i % 28 == 0){
      cout << "\n";
    }
  }

  cout << "\n";
  
  cout << labels.size() << endl; // how many labels you have
  cout << labels[0] << endl; // look at a label
}
