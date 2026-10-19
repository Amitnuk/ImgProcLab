#pragma once

#include <opencv2/opencv.hpp>
#include "math/Array3D.hpp"


namespace ImgProcLab::detail
{
    class GaussianData;
    
    GaussianData prepareGaussian(const uchar* pImage, 
                                 const float fSigma, 
                                 const std::size_t iKernelSize , 
                                 const std::size_t iHeight, 
                                 const std::size_t iWidth);

    void gaussianV1(GaussianData& eData);


    void create2DGaussianKernel(int iKernelSize, float fSigma);
}