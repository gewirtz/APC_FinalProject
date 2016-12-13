#include "reverse.h"

int Reverse(int i){

  // since data is imported in binary, we have to do
  // to bitwise operations!

  // bitwise operators modify variables considering
  // the bit patterns that represent the values
  // they store

  // & is bitwise addition
  // >> is shift bits to the right

  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;

  int reversed;

  reversed
    = ((int) ch1 << 24)
    + ((int) ch2 << 16)
    + ((int) ch3 << 8)
    + ch4;

  //cout << reversed << "\n";

  return reversed;

}
