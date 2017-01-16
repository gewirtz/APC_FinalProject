#include <cmath>
#include "matplotlibcpp.h"

// to compile: g++ plot_test.cpp -o test  -lpython2.7
//


using namespace std;
namespace plt = matplotlibcpp;



void plot_cost(std::vector<double> x, std::vector<double> y, const std::string outfile){
 plt::plot(x,y);
 plt::save(outfile);

}

int main(){


  int n=15;
  std::vector<double> x(n), y(n);
  for(int i=0; i<n; ++i) {
     x.at(i) = i;
     y.at(i) = i*i;
  }
  
  const std::string outfile = "test.png";
  plot_cost(x,y,outfile);
}


