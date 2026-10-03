#include <iostream>

template <typename t>

void printVector(t* vector, size_t N){
  for (int i = 0; i < N; i++){
    std::cout << i << " ";
  }
}
