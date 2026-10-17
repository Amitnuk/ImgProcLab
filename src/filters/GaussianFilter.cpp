#include <iostream>
#include <stdio.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "ImgProc/filters/GaussianFilter.hpp"
#include "ImgProc/cuda/GaussianCuda.hpp"

#define MAX_KERNEL_SIZE 15
using uchar = unsigned char;

namespace ImgProcLab
{
    namespace Filter{

        
        GaussianFilter::GaussianFilter(std::string_view sImgPath,
                                       std::size_t iKernelSize /*=3*/, 
                                       float fSigma /*= 1.0f*/, 
                                       GaussianVersion m_oVersion /*=GaussianVersion::V1*/)
        : m_iKernelSize(iKernelSize)
        //, m_iChannels(3)
        , m_oVersion(m_oVersion)
        , m_fSigma(fSigma)
        , m_aKernel(iKernelSize, iKernelSize, 1)
        {

            m_oInputImage = cv::imread(std::string(sImgPath));
            if(m_oInputImage.empty())
            {
                std::cout << sImgPath << std::endl;
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
            cv::imshow("Original", m_oInputImage);
            
            std::cout << "Gaussian Filter" << std::endl; 
            createGaussianKernel();
            switch(m_oVersion)
            {
                case GaussianVersion::V1:
                    detail::gaussianV1(m_oInputImage, m_oOutputImage, m_iKernelSize, m_fSigma);
                    break;
                
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
            cv::waitKey(0);
        }
    
        void GaussianFilter::createGaussianKernel()
        {   
    
            int iHalf = m_iKernelSize/2;
            float sum = 0.0f;
            for (int y = -iHalf; y <= iHalf; ++y)
            {
                for (int x = -iHalf; x <= iHalf; ++x)
                {
    
                    float fValue = (std::exp(-0.5f*(x*x + y*y)/(m_fSigma*m_fSigma)));
                    
                    m_aKernel(y+iHalf, x+iHalf) = fValue;
                    
                    sum += fValue;
                }
            }
    
            const float invSum = 1.0f / sum;
            for (size_t y = 0; y < m_iKernelSize; ++y)
            {
                for( size_t x = 0; x < m_iKernelSize; ++x )
                {
                    m_aKernel(y, x) *= invSum;
                }
            }
            
        }
    }

}
