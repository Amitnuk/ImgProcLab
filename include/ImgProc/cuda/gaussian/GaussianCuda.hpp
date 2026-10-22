#pragma once
#include <opencv2/opencv.hpp>
#include "common/ImgProcLabUtils.hpp"


enum class Separable
{
  TRUE,
  FALSE
};
  
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
                                   const Separable eCase = Separable::FALSE);

      void gaussianV1(GaussianData& eData, const int iThreads = DEFAULT_THREADS);
      void gaussianV2(GaussianData& eData, const int iThreads = DEFAULT_THREADS);

      void create2DGaussianKernel(float *pCoeficients,int iKernelSize, float fSigma);
      void create1DGaussianKernel(float *pCoeficients,int iKernelSize, float fSigma);
    }
}
