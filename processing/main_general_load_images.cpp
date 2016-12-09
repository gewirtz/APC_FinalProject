#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <assert.h>

// at the request of the modeling team...
#include <glob.h>

// at the request of the modeling team...
#include <armadillo>

//#include "general_load_images.h"

/*#include "loaded_data.h"
#include "reverse.cc"
#include "mnist_load_images.cc"
#include "mnist_load_labels.cc"
#include "mnist_count_images.cc"*/

using namespace std;
using namespace arma;

// Function to list files fom a given folder given the and extentsion or
// "*" which will check for all the files in that folder
// Written by Noemi
field<string> list_of_files(string path_images, string file_type){
	glob_t globbuf;
	path_images=path_images+file_type;
	//cout << path_images <<endl;

    int err = glob(path_images.c_str(), 0, NULL, &globbuf);
    if(err == 0){
    	int size = globbuf.gl_pathc;
    	field<string> list = field<string>(size);
        for (size_t i = 0; i < size; i++)  {
        	list(i)=globbuf.gl_pathv[i];
        }
        globfree(&globbuf);
        return(list);

    }

}

//int main(){
int main(int argc, char **argv) {


  // these will likely be taken from config file
  string images_directory = "../data/faces/training_faces/";
  string labels_directory = "../data/faces/training_faces_label/";
  string correct_or_wrong = "true"; // true for labeling images with 1 as correct images, and false to label as zero wrong images

  field<string> list_images = list_of_files(images_directory, "*");
  field<string> list_labels = list_of_files(labels_directory, "*");

  // check if exist a folder with the labes for that image or not
  if (list_labels.n_elem == 0){
  	// there is no label for this images, make label



  	cout << "stat3" <<endl;

  }



  

    return 0;



/*  // we are going to put things into an ARMADILLO MATRIX
  // which is really just a vector

  // load the images
  //vector<vector<double> > all_images;
  vector<arma::mat> all_images;
  mnist_load_images(train_directory, train_img, all_images);

  // return the number of images
  int num_img;
  num_img = mnist_count_images(train_directory,train_img, all_images);

  // load the labels
  //vector<double> labels(num_img);

  arma::colvec labels = arma::zeros<arma::colvec>(num_img);
  mnist_load_labels(train_directory, train_img, labels);


  // check to make sure you did it right
  
  cout << all_images.size() << endl; // how many images you have
  cout << all_images[0].size() << endl; // how big each image is
  
  // take a look at an image
  cout << all_images[0] << endl;

  cout << labels.size() << endl; // how many labels you have
  cout << labels[0] << endl; // look at a label
  */
}
