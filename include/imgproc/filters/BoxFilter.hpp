#pragma once
#include "imgproc/BaseFilter.hpp"

namespace ImgProc{
  class BoxFilter : public BaseFilter{
    
  public:
    explicit BoxFilter( int iWidth = 1, int iHeight = 32,int iChannels = 3,int iThreads = 16,int iKernelSize = 1);
    virtual ~BoxFilter() = default;
    virtual void KernelLauncher(const unsigned char* pIn_h, unsigned char* pOut_h) override;
    virtual void CPULauncher(const unsigned char* pIn_h, unsigned char* pOut_h) override {};
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


