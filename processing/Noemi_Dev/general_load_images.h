#ifndef GENERAL_LOAD_IMAGES_H_
#define GENERAL_LOAD_IMAGES_H_

using namespace std;
using namespace arma;


class Performance  {
 public:

  double mse(vec label, vec predict_label, int num_datapts); 
  mat correl(vec label, vec predict_label);
  double accuracy(vec label, vec predict_label);

 };

 bool fileExists(const string& filename);

#endif  // GENERAL_LOAD_IMAGES_H_
