#pragma once
#include <string_view>
#include <opencv2/opencv.hpp>
#include "ImgProc/cuda/gaussian/GaussianData.hpp"

namespace ImgProcLab { 

    
    enum class GaussianVersion
    {
        V1,
        V2,
        V3,
        V4,
        V5,
        V6
    };

    namespace Filter{
        
        class GaussianFilter 
        {   
            public :
                explicit GaussianFilter(const cv::Mat oImage,
                                        const std::size_t iKernelSize = 3, 
                                        const float fSigma = 1.0f, 
                                        const  GaussianVersion oVersion=GaussianVersion::V1);

                
                GaussianFilter(const GaussianFilter&) = delete;
                GaussianFilter& operator=(const GaussianFilter&) = delete;

                virtual ~GaussianFilter() = default;
                const detail::GaussianData& getGaussianData() const {return m_eGaussianData;}; 
                detail::GaussianData& getGaussianData() {return m_eGaussianData;}
                const cv::Mat getImage() const {return m_oInputImage;}
            private :
                
                std::size_t m_iKernelSize;
                //std::size_t m_iChannels;
                GaussianVersion m_eVersion;
                float m_fSigma;
                cv::Mat m_oInputImage; 
                cv::Mat m_oOutputImage;
                detail::GaussianData m_eGaussianData;
        };
    }

    
}


