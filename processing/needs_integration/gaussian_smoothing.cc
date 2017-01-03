/*
 * gaussian_smoothing.cc
 *
 *  Created on: Dec 26, 2016
 *      Author: ninagnedin
 */

// This is Gaussian_smoothing:process()
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <armadillo>
using namespace std;
using namespace arma;

int filter_size=5;
int bord=2;

//Filter components
float a0=0.0625;
float a1=0.25;
float a2=0.375;
float a3=0.25;
float a4=0.0625;



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
  //int c1=data_train[0].n_col;
  int c1=numcols;
  std::vector<float> conv_h0(c1);
  std::vector<float> conv_h1(c1);
  std::vector<float> conv_h2(c1);
  std::vector<float> conv_h3(c1);
  std::vector<float> conv_h4(c1);

   //Process training data
  //Vector_size is the # of images - do we feed this into the process function? or do we pass pointers to data?
  //for(int k=0;k<10;k++){
    //arma::mat currImg=data_train[k]; This is going to be angry - I don't have armadillo installed on my computer it's on the clusters
    //arma::mat currImg(10,10);
    //int r=currImg.n_rows;
    //int c=currImg.n_cols;
  int r=numrows;
  int c=numcols;
    arma::mat tempImg((r+2*bord),(c+2*bord)); //Larger array for mirror boundary conditions
    //Set boundaries
    //Set rows excluding corners
    for(int j=bord; j<(c+bord);j++){
	tempImg(0,j)=currImg(1,j-bord);
	tempImg(1,j)=currImg(0,j-bord);
	tempImg(r+bord+1,j)=currImg(r-2,j-bord);
	tempImg(r+bord,j)=currImg(r-1,j-bord);
    }
    cout<<"103\n";
    //Set cols excluding corners
    for(int i=bord; i<(r+bord);i++){
	 tempImg(i,0)=currImg(i-bord,1);
	 tempImg(i,1)=currImg(i-bord,0);
	 tempImg(i,c+bord)=currImg(i-bord,c-1);
	 tempImg(i,c+bord+1)=currImg(i-bord,c-2);
     }
    cout<<"108\n";
    //Corners - mirroring diagonally
    //Top left corner
    tempImg(0,1)=currImg(0,1);
    tempImg(1,0)=currImg(1,0);
    tempImg(0,0)=currImg(1,1);
    tempImg(1,1)=currImg(0,0);
   //Top right corner
    cout<<"117\n";
    cout<<"r "<<r<<"\n";
    cout<<"c "<<c<<"\n";
    cout<<"numrow temp img "<<tempImg.n_rows<<"\n";
    cout<<"numcol temp img "<<tempImg.n_cols<<"\n";
    cout<<"numrow c img "<<currImg.n_rows<<"\n";
    cout<<"numcol c img "<<currImg.n_cols<<"\n";
    tempImg(0,c+bord)=currImg(0,c-2);
    tempImg(1,c+bord)=currImg(0,c-1);
    tempImg(0,c+bord+1)=currImg(1,c-2);
    tempImg(1,c+bord+1)=currImg(1,c-1);
    //Bottom left corner
    cout<<"123";
    tempImg(r+bord,0)=currImg(r-2,0);
    tempImg(r+bord,1)=currImg(r-1,0);
    tempImg(r+2*bord-1,0)=currImg(r-2,1);
    tempImg(r+2*bord-1,1)=currImg(r-1,1);
    //Bottom right corner
    cout<<"129\n";
    tempImg(r+2*bord-1,c+2*bord-1)=currImg(r-2,c-2);
    tempImg(r+bord,c+2*bord-1)=currImg(r-2,c-1);
    tempImg(r+2*bord-1,c+bord)=currImg(r-1,c-2);
    tempImg(r+bord,c+bord)=currImg(r-1,c-1);

    // Now print the array to see the result
            for(row = 0; row < numrows+2*bord; ++row) {
              for(col = 0; col < numcols+2*bord; ++col) {
                printf("%2.0f ",tempImg(row,col));
              }
              printf("\n");
            }

    //Copy center of matrix
    cout<<"135\n";
    for(int i=bord;i<(r+bord);i++){
	for(int j=bord;j<(c+bord);j++){
	    tempImg(i,j)=currImg(i-bord,j-bord);
	}
    }

    // Now print the array to see the result
    for(row = 0; row < numrows+2*bord; ++row) {
               for(col = 0; col < numcols+2*bord; ++col) {
                 printf("%2.0f ",tempImg(row,col));
               }
               printf("\n");
             }

    //Sliding window - starting in top left corner
    //Horizontal convolution arrays

    cout<<"145\n";
    for(int j=bord;j<(c+bord);j++){ //Going across columns to get first 5 convolution arrays
	//conv_h0[j-bord]=a0*tempImg(0,j-2)+a1*tempImg(0,j-1)+a2*tempImg(0,j)+a3*tempImg(0,j+1)+a4*tempImg(0,j+2); //Two rows above
	conv_h1[j-bord]=a0*tempImg(0,j-2)+a1*tempImg(0,j-1)+a2*tempImg(0,j)+a3*tempImg(0,j+1)+a4*tempImg(0,j+2); //One row above
	conv_h2[j-bord]=a0*tempImg(1,j-2)+a1*tempImg(1,j-1)+a2*tempImg(1,j)+a3*tempImg(1,j+1)+a4*tempImg(1,j+2); //Current row
	conv_h3[j-bord]=a0*tempImg(2,j-2)+a1*tempImg(2,j-1)+a2*tempImg(2,j)+a3*tempImg(2,j+1)+a4*tempImg(2,j+2); //One row below
	conv_h4[j-bord]=a0*tempImg(3,j-2)+a1*tempImg(3,j-1)+a2*tempImg(3,j)+a3*tempImg(3,j+1)+a4*tempImg(3,j+2); //Two rows below
    }
    cout<<"159\n";
    for(int i=bord;i<(r+bord);i++){ //Vertical convolutions and moving down rows
	//Reset all horizontal convolutions and calc new one
	conv_h0=conv_h1;
	conv_h1=conv_h2;
	conv_h2=conv_h3;
	conv_h3=conv_h4;
	for(int j=bord;j<(c+bord);j++){ //Going across columns to get first 5 convolution arrays
		conv_h4[j-2]=a0*tempImg(i+2,j-2)+a1*tempImg(i+2,j-1)+a2*tempImg(i+2,j)+a3*tempImg(i+2,j+1)+a4*tempImg(i+2,j+2); //Two rows below
	}
	for(int j=bord;j<(c+bord);j++){
		    currImg(i-bord,j-bord)=a0*conv_h0[j-bord]+a1*conv_h1[j-bord]+a2*conv_h2[j-bord]+a3*conv_h3[j-bord]+a4*conv_h4[j-bord];
	}
    }

    // Now print the array to see the result
    for(row = 0; row < numrows; ++row) {
      for(col = 0; col < numcols; ++col) {
        printf("%2.0f ",currImg(row,col));
      }
      printf("\n");
    }

  }
