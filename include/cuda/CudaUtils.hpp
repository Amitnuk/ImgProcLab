#pragma once 

#define CUDA_CHECK( call )\
{\
  cudaError_t err = call;\
  if( err != cudaSuccess )\
  std::cerr << "Cuda Error " << err << "in " << __FILE__ <<":"<<__LINE__ << " : " << cudaGetErrorString(err) << " ( "<< #call << " ) " << std::endl ;\
}
