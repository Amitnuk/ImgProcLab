#pragma once
#pragma once
#include "ImgProc/core/BaseFilter.hpp"

namespace ImgProc{
  namespace color{
    
    template <typename T, typename U>
    class BGR2Gray : public BaseFilter<T,U>{
      
    public:
      explicit BGR2Gray( int iWidth = 1, int iHeight = 32,int iChannels = 3,int iThreads = 16);
      virtual ~BGR2Gray() = default;
      virtual void KernelLauncher(const T* pIn_h, U* pOut_h) override;
      virtual void CPULauncher(const T* pIn_h, U* pOut_h) override {};
    private:
      int m_iWidth;
      int m_iHeight;
      int m_iChannels;
      int m_iThreads;
      int m_iGridDimX;
      int m_iGridDimY;
      
      
    };  
  }
}