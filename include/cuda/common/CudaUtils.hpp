#pragma once 
#include <cstdio>


//TODO: create a inline function instead
#define CUDA_CHECK( call )\
{\
  cudaError_t err = call;\
  if( err != cudaSuccess )\
    std::fprintf(stderr,\
             "Cuda Error %d in %s:%d : %s ( %s )\n",\
             static_cast<int>(err),\
             __FILE__,\
             __LINE__,\
             cudaGetErrorString(err),\
             #call);\
}
