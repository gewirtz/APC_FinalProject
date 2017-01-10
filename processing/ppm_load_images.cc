#include <stdio.h>
#include <stdlib.h>
#include <armadillo>
#include <sys/stat.h>


using namespace std;
using namespace arma;

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

#define RGB_COMPONENT_COLOR 255

static PPMImage *readPPM(const char *filename){

         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

bool fileExists(const string& filename){
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1){
        return true;
    }
    return false;
}

void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    //fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

void grayscalePPM(PPMImage *img){
    int i;
    if(img){

         for(i=0;i<img->x*img->y;i++){

            //double gray = 0.2126 * img->data[i].red + 0.7152 * img->data[i].green + 0.0722 * img->data[i].blue;
            //double gray = (img->data[i].red+img->data[i].green+img->data[i].blue)/3;
            double gray = 0.2989*img->data[i].red+0.5870*img->data[i].green+0.1140*img->data[i].blue;

            img->data[i].red = gray;
            img->data[i].green = gray;
            img->data[i].blue = gray;           
         }
    }
}

vector<arma::mat> convertToArma(PPMImage *img){

    vector<arma::mat > all_images;

    //for(int i=0;i<img->x*img->y;i++){

    // create vector of doubles
    arma::mat cur_img(img->y,img->x);

    // all the images are the same size!
    // the images are stored in LITTLE-ENDIAN, no need
    // to reverse --> confusing!

    for(int row = 0; row < img->x; row++){
      for(int col = 0; col < img->y; col++){

        // adds pixel array
        cur_img(row,col) = (double) img->data[row*col].red;

      } // end col
    } // end row

    //puts current image at the end of the list
    all_images.push_back(cur_img);

    //} // end num_img
    return all_images;
}


// vector<mat> ppm_load_images(string directory, string filename){
//     PPMImage *image;
//     int i=0;
//     string fileout = "cars_000";
//     while(fileExists(fileout)){
//         fileout = filename + "_" + to_string(i);
//         readPPM(fileout);
//         cout << fileout <<endl;
//         fileout = "cars_000"
//         i++;
//     }

// }


int main(){
    PPMImage *image;
    image = readPPM("car_0001.ppm");


    //image = ppm_load_images("can_bottom.ppm");
    grayscalePPM(image);
    writePPM("test.ppm",image);
    printf("Press any key...");
    getchar();
}  