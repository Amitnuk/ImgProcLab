#ifndef BINARY_THRESHOLD_HPP
#define BINARY_THRESHOLD_HPP
#include "imgproc/BaseFilter.hpp"

namespace ImgProc {

  namespace Seg{ 
    template <typename T, typename U>
    class BinaryThreshold : public BaseFilter<T,U>
    {
      public: 
        explicit BinaryThreshold(std::size_t iWidth = 1, std::size_t iHeight = 32,int iThreads = 16, int iThreshold = 150 );
        virtual ~BinaryThreshold() = default;
        virtual void KernelLauncher(const T* pIn_h, U* pOut_h ) override;
        virtual void CPULauncher(const T* pIn_h, U* pOut_h )  override {};


      private:
        int m_iWidth;
        int m_iHeight;   
        int m_iThreads;
        int m_iThreshold;
        int m_iGridDimX;
        int m_iGridDimY;              
  };

 }

}





#endif
