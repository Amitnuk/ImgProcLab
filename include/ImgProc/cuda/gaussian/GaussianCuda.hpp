#pragma once
#include "common/ImgProcLabUtils.hpp"
#include "common/GaussianUtils.hpp"
#include <cstddef>

using uchar = unsigned char;

namespace ImgProcLab
{
    namespace detail
    {
      class GaussianData;
    
      GaussianData prepareGaussian(const uchar* pImage, 
                                   const float fSigma, 
                                   const std::size_t iKernelSize , 
                                   const std::size_t iHeight, 
                                   const std::size_t iWidth,
                                   const Separable eCase = Separable::FALSE,
                                   const METHOD eVersion = METHOD::NAIVE);

      void gaussianV1(GaussianData& eData, const int iThreads = CONSTANTS::DEFAULT_THREADS, const Separable eCase = Separable::FALSE);
      void gaussianV2(GaussianData& eData, const int iThreads = CONSTANTS::DEFAULT_THREADS, const Separable eCase = Separable::FALSE);
      void gaussianV3(GaussianData& eData, const int iThreads = CONSTANTS::DEFAULT_THREADS, const Separable eCase = Separable::FALSE); 

      void create2DGaussianKernel(float *pCoeficients,int iKernelSize, float fSigma);
      void create1DGaussianKernel(float *pCoeficients,int iKernelSize, float fSigma);
    }
}
