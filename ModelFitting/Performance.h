#ifndef PERFORMANCE_H_
#define PERFORMANCE_H_

using namespace std;
using namespace arma;


class Performance  {
 public:

  double mse(vec label, vec predict_label, int num_datapts); 
  mat correl(vec label, vec predict_label);
  double accuracy(vec label, vec predict_label);

 };

#endif  // PERFORMANCE_H_
