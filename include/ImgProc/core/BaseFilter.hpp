#pragma once

namespace ImgProc{
  template <typename T, typename U>
  class BaseFilter {
    
  protected:
    virtual ~BaseFilter() = default;
    virtual void KernelLauncher(const T* pIn_h, U* pOut_h ) = 0;
    virtual void CPULauncher(const T* pIn_h, U* pOut_h ) = 0; 
  };  
}


