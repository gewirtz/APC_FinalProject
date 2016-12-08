#ifndef PERFORMANCE_H_
#define PERFORMANCE_H_



class Performance  {
 public:

  arma::vec mse(double *label, double *predict_label, int num_datapts); //gives exact solution 



 };

#endif  // PERFORMANCE_H_
