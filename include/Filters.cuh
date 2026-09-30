#pragma once

namespace Filters{
  class AverageFilter {
    
  public:
    AverageFilter(
      int iWidth = 1,
		  int iHeight = 32,
      int iChannels = 3,
      int iThreads = 16,
		  int iKernelSize = 3);

    ~AverageFilter() = default;
    void KernelLauncher(const unsigned char* pIn_h, unsigned char* pOut_h);
    
    
  private:
    int m_iWidth;
    int m_iHeight;
    int m_iChannels;
    int m_iThreads;
    int m_iKernelSize;
    int m_iGridDimX;
    int m_iGridDimY;
    
    
  };  
}


