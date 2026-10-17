#ifndef GAUSSIAN_FILTER
#define GAUSSIAN_FILTER 
#include <string_view>
#include <opencv2/opencv.hpp>
#include "math/Array3D.hpp"

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
                explicit GaussianFilter( std::string_view sImgPath,
                                         std::size_t iKernelSize = 3, 
                                         float fSigma = 1.0f, 
                                         GaussianVersion oVersion=GaussianVersion::V1);

                virtual ~GaussianFilter() = default;
            private :
                
                std::size_t m_iKernelSize;
                //std::size_t m_iChannels;
                GaussianVersion m_oVersion;
                float m_fSigma;
                cv::Mat m_oInputImage; 
                cv::Mat m_oOutputImage;
                Array<float> m_aKernel;
    
                void createGaussianKernel();
          
        };
    }

    
}


#endif 