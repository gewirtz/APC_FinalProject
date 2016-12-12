#include <iostream>
#include <fstream>
#include <armadillo>

using namespace std;
using namespace arma;

void mnist_load_images(string directory, string filename,
		       vector<arma::mat > &all_images){

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
    arma::mat cur_img(rows,cols);

    // all the images are the same size!
    // the images are stored in LITTLE-ENDIAN, no need
    // to reverse --> confusing!
    for(int row = 0; row < rows; row++){
      for(int col = 0; col < cols; col++){

        unsigned char pixel = 0; // 1 byte
        mnist_image.read( (char*) &pixel, sizeof(pixel) );

        // adds pixel array
        cur_img(row,col) = (double) pixel;

      } // end col
    } // end row

    //puts current image at the end of the list
    all_images.push_back(cur_img);

  } // end num_img

} // end function load_mnist

