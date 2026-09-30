#include <iostream>
#include "Filters.cuh"



__global__ void AverageFilter(unsigned char* in, unsigned char* out)
{
  
}

namespace Filters {
  
 
  AverageFilter::AverageFilter(int iGridDim,
			       int iBlockDim,
			       int iKernelSize)
    : m_iGridDim(iGridDim)
    , m_iBlockDim(iBlockDim)
    , m_iKernelSize(iKernelSize)
  {
    std::cout << "Average Filter" << std::endl; 
  }
  
  void AverageFilter::KernelLauncher(unsigned char* pIn, unsigned char* pOut)
  {
    std::cout << "Kernel Launcher" << std::endl;
  }
}
