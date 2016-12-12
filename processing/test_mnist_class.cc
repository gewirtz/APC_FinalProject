#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_load_mnist.h"

int main()

{

data_process_base *mnist_data;
data_process_base *mnist_labels;

  string train_directory = "../data/mnist/training/";
  string test_directory = "../data/mnist/testing/";

  string train_lbl = "train-labels.idx1-ubyte";
  string train_img = "train-images.idx3-ubyte";
  string test_lbl = "t10k-labels.idx1-ubyte";
  string test_img = "t10k-images.idx3-ubyte";

mnist_data = new data_load_mnist(train_directory, train_img);
mnist_labels = new data_load_mnist(train_directory, train_lbl);

// then what?

}