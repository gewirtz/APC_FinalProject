#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "GradientModel.h"
#include "KNN.h"

class Model;


class Optimizer {
 public:
  virtual ~Optimizer() {}
  virtual void fitParams(Model *m) = 0;
  virtual void fitParams(KNN *m) = 0;
  virtual void fitParams(GradientModel *m) = 0;

};

#endif  // OPTIMIZER_H_
