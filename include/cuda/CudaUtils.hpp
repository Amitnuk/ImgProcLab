#pragma once 

#define CUDA_CALL( call )\
{\
  cudaError_t err = call;\
  if( err != cudaSuccess )\
  std::cerr << "Cuda Error " << err << "in " << __FILE__ <<":"<<__LINE__ << " : " << cudaGetErrorString(err) << " ( "<< #call << " ) " << std::endl ;\
}
