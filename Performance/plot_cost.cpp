//
//  Plot Cost Function
//
//  Noemi Vergopolan
//


#include <cmath>
#include "matplotlibcpp.h"
#include <armadillo>

using namespace std;
using namespace arma;

namespace plt = matplotlibcpp;

//std::vector<std::vector<double>> costs;
//std::vector<double> index(costs[0].size());
//std::vector<int> labels(costs.size());

void plot_cost(vector<vector<double>> cost, const std::string outfile){
  
  int ns = cost.size();
  for(int s=0; s<ns; ++s ){
    int n=cost[s].size();
    std::vector<double> x(n), y(n);
    for(int i=0; i<n; ++i) {
      x.at(i) = i;
      y.at(i) = cost[s][i];
    } 
    plt::plot(x,y);
  }
  plt::title("Gradient Descent");
  plt::xlabel("Iteration number");
  plt::xlabel("Cost");
  plt::save(outfile); 
     
}


