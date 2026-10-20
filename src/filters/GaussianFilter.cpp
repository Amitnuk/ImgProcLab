#include <iostream>
#include <stdio.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "ImgProc/cuda/gaussian/GaussianCuda.hpp"
#include "ImgProc/filters/GaussianFilter.hpp"

#define MAX_KERNEL_SIZE 15
using uchar = unsigned char;

namespace ImgProcLab
{
    namespace Filter{

        
        GaussianFilter::GaussianFilter(const cv::Mat oImage,
                                       const std::size_t iKernelSize /*=3*/, 
                                       const float fSigma /*= 1.0f*/, 
                                       const GaussianVersion eVersion /*=GaussianVersion::V1*/)
        : m_iKernelSize(iKernelSize)
        //, m_iChannels(3)
        , m_eVersion(eVersion)
        , m_fSigma(fSigma)
        , m_oInputImage(oImage)
        {

            if(m_oInputImage.empty())
            {
                std::cout << m_oInputImage << std::endl;
                throw std::invalid_argument("Input Image is empty");
            }
        
            if( m_oInputImage.type() != CV_8UC3 )
            {
                throw std::invalid_argument("Gaussian filter only supporst CV_8UC3 as image type");
            }
            
            if( m_iKernelSize%2 == 0)
            {
                std::cout << "[WARNING] : A KERNEL SIZE SHOULD BE A ODD NUMBER SUP THAN 1, eg 3, 5, 7, 9, ..., 2*n+1" << std::endl;
            }

            if( m_fSigma <= 0.0f )
            {
                throw std::invalid_argument("Gaussian sigma must be positive");
            }
            
            auto [iWidth, iHeight] = oImage.size();
            std::cout << "Gaussian Filter" << std::endl; 
            std::cout << "(height, width)="<< "("<< iHeight << "," << iWidth<<")"<< std::endl;
            std::cout << "(height, width)="<< "("<< m_oInputImage.rows << "," << m_oInputImage.cols<<")"<< std::endl;
            

            switch(m_eVersion)
            {
                case GaussianVersion::V1:
                {
                    uchar* pImage = m_oInputImage.ptr<uchar>();
                    m_eGaussianData = detail::prepareGaussian(pImage, fSigma, iKernelSize, iHeight, iWidth);
                    break;

                }
                case GaussianVersion::V2:
                    break;
                
                case GaussianVersion::V3:
                    break;

                case GaussianVersion::V4:
                    break;

                case GaussianVersion::V5:
                    break;
                
                case GaussianVersion::V6:
                    break;
                default:
                    break;
            }
            
        }
    }

}
