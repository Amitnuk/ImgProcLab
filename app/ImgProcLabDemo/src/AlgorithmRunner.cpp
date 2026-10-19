#include <opencv2/opencv.hpp>
#include "AlgorithmRunner.hpp"
#include "ImgProc/cuda/gaussian/GaussianData.hpp"
#include "ImgProc/filters/GaussianFilter.hpp"


void AlgorithmRunner::prepare()
{

    auto displayInfo = [](const AlgorithmConfig& eConfig) 
    {
        if(eConfig.m_iKernelSize)
        {
            std::cout << "Kernel size: " << *eConfig.m_iKernelSize << std::endl;
        }

        if(eConfig.m_iHeight)
        {
            std::cout << "Height: " << *eConfig.m_iHeight << std::endl;
        }

        if(eConfig.m_iWidth)
        {
            std::cout << "Width: " << *eConfig.m_iWidth << std::endl;
        }

        if(eConfig.m_fSigma)
        {
            std::cout << "Sigma: " << *eConfig.m_fSigma << std::endl;
        }
        if(!(*eConfig.m_sImagePath).empty())
        {
            std::cout << "ImagePath: " << *eConfig.m_sImagePath << std::endl;
        }
        
    };

    switch(*m_eConfig.m_eAlgorithmType)
    {
        case Algorithm::GaussianFilter:
            switch(*m_eConfig.m_eVersion)
            {
                case Version::V1:
                    {
                        std::cout << "Executing GaussianFilter V1" << std::endl;
                        displayInfo(m_eConfig);
                        cv::Mat oImage = cv::imread(std::string(*m_eConfig.m_sImagePath));
                        ImgProcLab::Filter::GaussianFilter oGaussianFilter( oImage, *m_eConfig.m_iKernelSize, *m_eConfig.m_fSigma, ImgProcLab::GaussianVersion::V1);

                        m_eGaussianData =  std::move(oGaussianFilter.getGaussianData());
                        cv::imshow("Original", oImage);
                        ImgProcLab::detail::gaussianV1(m_eGaussianData);
                        int iHeight  = m_eGaussianData.m_eCudaExecutionContext.m_iHeight;   
                        int iWidth   = m_eGaussianData.m_eCudaExecutionContext.m_iWidth; 
                        uchar* pOut = new uchar[iWidth*iHeight*3]; 
                        
                        m_eGaussianData.m_eCudaExecutionContext.m_oDeviceOutputs.copyFromDevice(pOut);
                        cv::Mat oOutputImage(iHeight, iWidth, CV_8UC3, pOut);
                        
                        cv::imshow("GaussianFilter", oOutputImage);
                        
                        cv::waitKey(0);
                        delete[] pOut;
                        break;
                    }
                    case Version::V2:
                    {
                        std::cout << "Executing GaussianFilter V2" << std::endl;
                        break;
                    }            
                    default:
                    {
                        throw std::invalid_argument("[ERROR]: Executing GaussianFilter with Unknown Version");
                    }
                    break;
            }
            break;
        default:
        {
            throw std::invalid_argument("Executing an Unknonw Algorithm");
        }
                
            
        
    }
}



void AlgorithmRunner::execute()
{

}

void AlgorithmRunner::run()
{
    prepare();

}