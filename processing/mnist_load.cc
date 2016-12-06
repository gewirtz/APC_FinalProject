#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

// heavily inspired by "canonical" mnist data readers

using namespace std;

void load_mnist(string filename){


  ifstream ifs (filename.c_str(), ifstream::in);
  char c = ifs.get();

  while(ifs.good()){
    cout <<c;
    c = ifs.get();
  }

  ifs.close();

}


int main(){

  string train_lbl = "../data/mnist/training/train-labels.idx1-ubyte";
  string train_img = "../data/mnist/training/train-images.idx3-ubyte";
  string test_lbl = "../data/mnist/testing/t10k-labels.idx1-ubyte";
  string test_img = "../data/mnist/testing/t10k-images.idx3-ubyte";

  load_mnist(filename);

  //vector<vector<double>> vec;

  //mnist_load(filename,vec);

  //cout << filename << "\n";

  return 0;
}
