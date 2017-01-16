#include <stdio.h>
#include <stdlib.h>
#include <armadillo>
#include <sys/stat.h>
//#include "jpg_load_images.h"
#include "CImg.h"


using namespace std;
using namespace arma;
using namespace cimg_library;

typedef struct {
     unsigned char red,green,blue;
} JPGPixel;

typedef struct {
     int x, y;
     JPGPixel *data;
} JPGImage;

arma::mat readJPG(const char *img){

    CImg<unsigned char> src(img);
    int width = src.width();
    int height = src.height();
    arma::mat cur_img(height,width);
    //cout << width << "x" << height << endl;
    for (int r = 0; r < height; r++){
        for (int c = 0; c < width; c++){
             cur_img(r,c) = 0.2989*(int)src(c,r,0,0)+0.5870*(int)src(c,r,0,1)+0.1140*(int)src(c,r,0,2);
            //For printing greyscale values
            //cout << "(" << r << "," << c << ") ="
            //     << cur_img(r,c) << endl;
         }
    }
    return cur_img;

}

vector<string> fileNames(const char* direc){
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(direc)) !=NULL){
        vector<string> f_names;
      
        while ((ent = readdir(dir)) !=NULL){
            //printf("%s\n",ent->d_name);
            f_names.push_back(ent->d_name);
        }
        closedir(dir);
        return f_names;
    } //else{
      // perror("");
      //  return EXIT_FAILURE;
    //}
    
}

vector<arma::mat> jpg_load_images(string testing_data){
    //string testing_data = "/home/andreas/APC524/Project/data/faces/testing_faces/";
    vector<string> fnames;
    fnames = fileNames(testing_data.c_str());

    vector<arma::mat> all_images;

    for(int i=2;i<fnames.size();i++){
        //cout << (string) fnames[i] << endl;
        arma::mat image;
        //cout << testing_data + fnames[i].c_str() <<endl;
        image = readJPG((testing_data + fnames[i]).c_str());
        
        //convertToArma(image);
        all_images.push_back(image);
        //cout << fnames[3] << endl;
    }
    return all_images;

// int main(){



// }