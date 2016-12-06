#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

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
void load_mnist(string directory, string filename){

  string fname = directory + filename;
  cout << fname << "\n";

  // the example I found imports data in binary:
  ifstream mnist_file (fname.c_str(), ios::binary);
  
  if(mnist_file.is_open()){
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
    mnist_file.read( (char*) &magic, sizeof(magic) );
    Reverse(magic);

    // Next is the number of images
    mnist_file.read( (char*) &num_img, sizeof(num_image) );
    Reverse(num_img);

    // rows, cols
    mnist_file.read( (char*) &rows, sizeof(rows) );
    mnist_file.read( (char*) &cols, sizeof(cols) );
    Reverse(rows);
    Reverse(cols);

    // The rest of the data is the image data
    // the individual images are stored in chunks of 32 bits/
    // We need to RESHAPE them from 1D array -> 2D array
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

  load_mnist(train_directory, train_img);

  //vector<vector<double>> vec;

  //mnist_load(filename,vec);

  //cout << filename << "\n";

  return 0;
}
