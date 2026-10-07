#pragma once
#include "imgproc/BaseFilter.hpp"

namespace ImgProc{
  namespace Blur{
    
    template <typename T, typename U>
    class MedianFilter : public BaseFilter<T,U>{
      
    public:
      explicit MedianFilter( int iWidth = 1, int iHeight = 32,int iChannels = 3,int iThreads = 16,int iKernelSize = 1);
      virtual ~MedianFilter() = default;
      virtual void KernelLauncher(const T* pIn_h, U* pOut_h) override;
      virtual void CPULauncher(const T* pIn_h, U* pOut_h) override {};
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
}

