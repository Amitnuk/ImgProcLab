#pragma once

#include <opencv2/opencv.hpp>
#include "math/Array3D.hpp"


namespace ImgProcLab::detail
{
    void gaussianV1(const cv::Mat& oInputImage,  
                    cv::Mat& oOutputImage, 
                    const std::size_t& iKernelSize=3,  
                    const float& fSigma=1.0f, 
                    const std::size_t& iChannels=3);


    Array<float> create2DGaussianKernel(int iKernelSize, float fSigma);
}