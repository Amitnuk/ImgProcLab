#pragma once

namespace Filters{
  class AverageFilter {
    
  public:
    AverageFilter(int iGridDim = 1,
		  int iBlockDim = 32,
		  int iKernelSize = 3);
    ~AverageFilter() = default;
    void KernelLauncher(unsigned char* pIn, unsigned char* pOut);
    
  private:
    int m_iGridDim;
    int m_iBlockDim;
    int m_iKernelSize;
  };  
}


