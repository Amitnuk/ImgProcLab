#ifndef SOBEL_FILTER_HPP
#define SOBEL_FILTER_HPP
#include "imgproc/BaseFilter.hpp"
#include "math/Array3D.hpp"

namespace ImgProc {

  namespace Edge{ 
    template <typename T, typename U>
    class SobelFilter : public BaseFilter<T,U>
    {
      public: 
        explicit SobelFilter(std::size_t iWidth = 1, std::size_t iHeight = 32,int iThreads = 16, std::size_t iKernelSize = 3);
        virtual ~SobelFilter() = default;
        virtual void KernelLauncher(const T* pIn_h, U* pOut_h ) override;
        virtual void CPULauncher(const T* pIn_h, U* pOut_h )  override {};


      private:
        int m_iWidth;
        int m_iHeight;   
        int m_iThreads;
        std::size_t m_iKernelSize;
        float m_fNormalizingFactor;
        Array<int> m_aKernelX;
        Array<int> m_aKernelY;
        int m_iGridDimX;
        int m_iGridDimY;      
        

        void createSobelKernel();
        
  };

 }

}





#endif
