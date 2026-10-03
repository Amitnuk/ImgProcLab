#ifndef GAUSSIAN_FILTER
#define GAUSSIAN_FILTER 
#include "imgproc/BaseFilter.hpp"
#include "math/Array3D.hpp"


namespace ImgProc { 
    class GaussianFilter : public BaseFilter
    {   
        public :
            explicit GaussianFilter(std::size_t iWidth = 1, std::size_t iHeight = 32,std::size_t iChannels = 3,int iThreads = 16, std::size_t iKernelSize = 3, float fSigma = 1.0f);
            virtual ~GaussianFilter() = default;
            virtual void KernelLauncher(const unsigned char* pIn_h, unsigned char* pOut_h ) override;
            virtual void CPULauncher(const unsigned char* pIn_h, unsigned char* pOut_h ) override {}; 
    
        private :
            int m_iWidth;
            int m_iChannels;
            int m_iHeight;
            int m_iThreads;
            int m_iGridDimX;
            int m_iGridDimY;
            std::size_t m_iKernelSize;
            std::size_t m_fSigma;
            Array<float> m_aKernel;

            void createGaussianKernel();
      
    };
}


#endif 