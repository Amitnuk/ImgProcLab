#pragma once
#include <string_view>
#include <opencv2/opencv.hpp>
#include "ImgProc/cuda/gaussian/GaussianData.hpp"
#include "common/GaussianUtils.hpp"

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
                                        const  GaussianVersion oVersion=GaussianVersion::V1,
                                        const Separable eCase = Separable::FALSE,
                                        const METHOD eMethod = METHOD::NAIVE);

                
                GaussianFilter(const GaussianFilter&) = delete;
                GaussianFilter& operator=(const GaussianFilter&) = delete;

                GaussianFilter(GaussianFilter&& oRhs) noexcept
                : m_iKernelSize(oRhs.m_iKernelSize)
                , m_eVersion(oRhs.m_eVersion)
                , m_fSigma(oRhs.m_fSigma)
                , m_oInputImage(std::move(oRhs.m_oInputImage))
                , m_oOutputImage(std::move(oRhs.m_oOutputImage))
                , m_eGaussianData(std::move(oRhs.m_eGaussianData))
                {
              
                }

                GaussianFilter& operator=(GaussianFilter&& oRhs) noexcept
                {
                  if(this == &oRhs)
                  {
                    return *this;
                  }

                  m_iKernelSize = oRhs.m_iKernelSize;
                  m_eVersion = oRhs.m_eVersion;
                  m_fSigma = oRhs.m_fSigma;
                  m_oInputImage = std::move(oRhs.m_oInputImage);
                  m_oOutputImage = std::move(oRhs.m_oOutputImage);
                  m_eGaussianData =std::move(oRhs.m_eGaussianData);

                  return *this;
                  
                }   
          
                ~GaussianFilter() = default;
                //TODO: 
                // oGaussianFilter..getGaussianData() performs a move so oGaussianFilter does not have it anymore, should allow copy, Array should be copiable

                const detail::GaussianData& getGaussianData() const {return std::move(m_eGaussianData);}; // const lvalue can be initialized with rvalue(1)
                detail::GaussianData&& getGaussianData() {return std::move(m_eGaussianData);} // rvalue can be initialized with rvalue(2)
            // const detail::GaussianData& getGaussianData() {return m_eGaussianData;}; //const lvalue can be initialized with lvalue(3) 
                const cv::Mat getImage() const {return m_oInputImage;}
            private :
                
                std::size_t m_iKernelSize;
                //std::size_t m_iChannels;
                GaussianVersion m_eVersion;
                Separable m_eCase;
                METHOD m_eMethod;
                float m_fSigma;
                cv::Mat m_oInputImage; 
                cv::Mat m_oOutputImage;
                detail::GaussianData m_eGaussianData;
        };
    }

    
}


