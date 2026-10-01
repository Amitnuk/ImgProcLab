#pragma once

namespace ImgProc{
  class BaseFilter {
    
  protected:
    virtual ~BaseFilter() = default;
    virtual void KernelLauncher(const unsigned char* pIn_h, unsigned char* pOut_h ) = 0;
    virtual void CPULauncher(const unsigned char* pIn_h, unsigned char* pOut_h ) = 0; 
  };  
}


