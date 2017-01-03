/*
 * histogram.cc
 *
 *  Created on: Jan 3, 2017
 *      Author: ninagnedin
 */


// This is histogram:process()
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <armadillo>
using namespace std;
using namespace arma;

int main(){
  //Read in image
  /*FILE *pFile; */

 /*pFile=fopen("a.pgm","r+");
  fscanf(pFile,"i");
  fprintf(pFile,"i");
  for(int row=0;row<2;row++){
      for(int col=0;col<2;col++){
	  unsigned char pixel=0;
	  pixel=fgetc(pFile);
	  currImg(row,col)=pixel;
	  cout<<pixel;
      }
  }*/

  int row = 0, col = 0, numrows = 0, numcols = 0;
    ifstream infile("a.pgm");
    stringstream ss;
    string inputLine = "";

    // First line : version
    getline(infile,inputLine);
    if(inputLine.compare("P2") != 0) cerr << "Version error" << endl;
    else cout << "Version : " << inputLine << endl;

    // Second line : comment
    getline(infile,inputLine);
    cout << "Comment : " << inputLine << endl;

    // Continue with a stringstream
    ss << infile.rdbuf();
    // Third line : size
    ss >> numcols >> numrows;
    cout << numcols << " columns and " << numrows << " rows" << endl;
    //Current image
    arma::mat currImg(numrows,numcols);
    //Fourth line: max mag
    int maxval;
    ss >> maxval;
    // Following lines : data
    for(row = 0; row < numrows; ++row)
      for (col = 0; col < numcols; ++col) ss >> currImg(row,col);

    // Now print the array to see the result
    for(row = 0; row < numrows; ++row) {
      for(col = 0; col < numcols; ++col) {
        printf("%2.0f ",currImg(row,col));
      }
      printf("\n");
    }
    infile.close();

  // Initialize arrays - test and train images have same dimensions
  //# of pixel options won't change
  int pixels=256;
  //when we connect this to the overall system k will be the number of images - for now just using 1 bc have 1 test image
  int n =1;
  //Matrix where each row is a histogram for one image - initialize w all zeros
  arma::mat hists(n,pixels,fill::zeros);
  //arma::mat currImg();
   //When we have k images we will add another outer loop here to iterate through those
  //for(int k=0;k<n,k++){
  //	currImg=data[k]; where data is data_train or data_test depending on input
   for(int i=0;i<numrows;i++){
       for(int j=0;j<numcols;j++){
	   int temp_pixel=currImg(i,j);
	   printf("%d",temp_pixel);
	   printf("\n");
	   //hists(k,temp_pixel)++;
	   hists(0,temp_pixel)++;
       }
   }
//}

    // Now print the array to see the result
    for(row = 0; row < 1; ++row) {
      for(col = 0; col < pixels; ++col) {
        printf("%2.0f ",hists(row,col));
      }
      printf("\n");
    }

  }


