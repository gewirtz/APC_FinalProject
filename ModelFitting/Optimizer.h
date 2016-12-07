#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "model.h"

class Model;

class Optimizer {
 public:
  virtual ~Optimizer() {}
  virtual void fitParams(Model* m) = 0;

};

#endif  // OPTIMIZER_H_
