#include "ImgProc/basicOperations/TransformColor.hpp"
#include "ImgProc/cuda/basicOperations/colorConversion/iplCuda.hpp"
#include "ImgProc/cuda/basicOperations/colorConversion/iplData.hpp"
#include <cmath>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <stdio.h>
using uchar = unsigned char;


namespace ImgProcLab
{
    namespace basicOps
    {

        rgb2gray::rgb2gray(const cv::Mat oImage)
        : m_oInputImage(oImage)
        {

            if(m_oInputImage.empty())
            {
                std::cout << m_oInputImage << std::endl;
                throw std::runtime_error("Input Image is empty");
            }

            if(m_oInputImage.type() != CV_8UC3)
            {
                throw std::runtime_error("Only support CV_8UC3 as image type");
            }


            auto [iWidth, iHeight] = oImage.size();
            std::cout << "Gaussian Filter" << std::endl;
            std::cout << "(height, width)="
                      << "(" << iHeight << "," << iWidth << ")" << std::endl;
            std::cout << "(height, width)="
                      << "(" << m_oInputImage.rows << "," << m_oInputImage.cols << ")" << std::endl;

            uchar* pImage = m_oInputImage.ptr<uchar>();
            m_eColorConversionData = detail::prepareGrayConversion(pImage, iHeight, iWidth);
        }
    } // namespace basicOps

} // namespace ImgProcLab
