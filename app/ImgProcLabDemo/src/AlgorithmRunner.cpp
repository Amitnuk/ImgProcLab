#include <opencv2/opencv.hpp>
#include "AlgorithmRunner.hpp"
#include "BenchmarkWriter.hpp"
#include <opencv2/cudafilters.hpp> 
#include <ImgProc/cuda/gaussian/GaussianData.hpp>
#include <ImgProc/filters/GaussianFilter.hpp>
#include <cuda/common/CudaExecution.hpp>
#include <cuda/common/OpenCVExecution.hpp>
#include "config.h"


namespace 
{
    void compareImage(const cv::Mat& oAtual, const cv::Mat& oReference)
    {
        if(oAtual.size() != oReference.size() || oAtual.type() != oReference.type())
        {
            throw std::invalid_argument("Selected images are incompatible");
        }
        
        cv::Mat oDifference;
        cv::absdiff(oAtual, oReference, oDifference);
        cv::Scalar oMean = cv::mean(oDifference);
        float fMAE = 0.0f;
        const size_t iChannels = oAtual.channels();
        for(size_t i = 0; i < iChannels; ++i)
        {
            fMAE += oMean[i];
        }
        
        fMAE /=iChannels;
        double fMaxError = 0.0;
        cv::minMaxLoc(oDifference.reshape(1), nullptr, &fMaxError);

        std::cout << "Mean Absolute Error: " << fMAE << std::endl;
        std::cout << "Maximum Absolute Error: " << fMaxError << std::endl;
    }

    auto displayInfo = [](const AlgorithmConfig &eConfig)
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
} // namespace 

void AlgorithmRunner::prepare()
{

    

    switch(*m_eConfig.m_eAlgorithmType)
    {
        case Algorithm::GaussianFilter:
        {
            switch(*m_eConfig.m_eVersion)
            {
                case Version::V1:
                {
                    std::cout << "Executing GaussianFilter V1" << std::endl;
                    std::cout << "Gaussian Kernel Launcher" << std::endl;

                    ::displayInfo(m_eConfig);
                    cv::Mat oImage = cv::imread(std::string(*m_eConfig.m_sImagePath));
                    ImgProcLab::Filter::GaussianFilter oGaussianFilter( oImage,
                                                                        *m_eConfig.m_iKernelSize,
                                                                        *m_eConfig.m_fSigma,
                                                                        ImgProcLab::GaussianVersion::V1);
                            
                    m_eGaussianData =  std::move(oGaussianFilter.getGaussianData());
                    
                    float fElapsedTime = executeCuda([&](){ ImgProcLab::detail::gaussianV1(m_eGaussianData);}, 
                                                     m_eConfig.m_bBenchmark, 
                                                     m_eConfig.m_iGPUWarmUps, 
                                                     m_eConfig.m_iIterations);

                    int iHeight   = m_eGaussianData.m_eCudaExecutionContext.m_iHeight;   
                    int iWidth    = m_eGaussianData.m_eCudaExecutionContext.m_iWidth;
                    assert(oImage.rows == iHeight);
                    assert(oImage.cols == iWidth);

                    cv::Mat oOutputImageCPU;
                    cv::Mat oOutputImageCUDA;
                    cv::Mat oOutputImage;                    
                    if(m_eConfig.m_bBenchmark)
                    {
                        const std::string& sPath = GetBenchmarkDir() + "gaussianV1.csv";
                        BenchmarResult oBenchmarResult{iWidth,iHeight,
                            *m_eConfig.m_iKernelSize,
                            m_eConfig.m_iGPUWarmUps,
                            m_eConfig.m_iIterations,
                            fElapsedTime,
                            "gaussian",
                            "v1"};  
                        
                        BenchmarkWriter::writeCSV(sPath, oBenchmarResult);

                        float fElapsedTImeCPU = executeOpenCV([&](){cv::GaussianBlur(oImage,
                                                                                     oOutputImageCPU, 
                                                                                     cv::Size(*m_eConfig.m_iKernelSize,*m_eConfig.m_iKernelSize), 
                                                                                     *m_eConfig.m_fSigma,
                                                                                     *m_eConfig.m_fSigma,
                                                                                     cv::BORDER_CONSTANT);}, 
                                                                    Backend::CPU, 
                                                                    m_eConfig.m_bBenchmark,
                                                                    m_eConfig.m_iGPUWarmUps,
                                                                    m_eConfig.m_iIterations);



                        BenchmarResult oBenchmarResultCPU{iWidth,
                                                          iHeight,    
                                                          *m_eConfig.m_iKernelSize,   
                                                          m_eConfig.m_iGPUWarmUps,
                                                          m_eConfig.m_iIterations,
                                                          fElapsedTImeCPU,
                                                          "gaussian",
                                                          "CPU"};
            
                        BenchmarkWriter::writeCSV(sPath, oBenchmarResultCPU);


                            
                            
                        cv::cuda::GpuMat d_input, d_output;
                        d_input.upload(oImage);
                        cv::cuda::Stream stream;
                        auto oGaussian = cv::cuda::createGaussianFilter(oImage.type(), 
                                                                        oImage.type(),
                                                                        cv::Size(*m_eConfig.m_iKernelSize, *m_eConfig.m_iKernelSize), 
                                                                        *m_eConfig.m_fSigma, 
                                                                        *m_eConfig.m_fSigma,
                                                                        cv::BORDER_CONSTANT);
                                                                        
                        float fElapsedTImeCUDA = executeOpenCV([&](){oGaussian->apply(d_input, d_output, stream);},
                                                                    Backend::CUDA, 
                                                                    m_eConfig.m_bBenchmark, 
                                                                    m_eConfig.m_iGPUWarmUps,
                                                                    m_eConfig.m_iIterations);

                        d_output.download(oOutputImageCUDA);
                        BenchmarResult oBenchmarResultCUDA{iWidth,
                            iHeight,
                            *m_eConfig.m_iKernelSize,
                            m_eConfig.m_iGPUWarmUps, 
                            m_eConfig.m_iIterations, 
                            fElapsedTImeCUDA,
                            "gaussian", 
                            "CUDA"};

                        BenchmarkWriter::writeCSV(sPath, oBenchmarResultCUDA);
                        
                        bool bVisualize = true;
                        if( bVisualize )
                        {
                            cv::imshow("Original", oImage);
                            cv::imshow("GaussianFilterCPU", oOutputImageCPU);
                            cv::imshow("GaussianFilterCUDA", oOutputImageCUDA);


                            constexpr int iChannels = 3;
                            uchar* pOut = new uchar[iWidth*iHeight*iChannels]; 
                                
                            m_eGaussianData.m_eCudaExecutionContext.m_oDeviceOutputs.copyFromDevice(pOut);
                            oOutputImage = cv::Mat(iHeight, iWidth, CV_8UC3, pOut);
                

                            int radius = *m_eConfig.m_iKernelSize / 2;

                            cv::Rect roi(
                                radius,
                                radius,
                                oOutputImage.cols - 2 * radius,
                                oOutputImage.rows - 2 * radius
                            );

                            std::cout << "OpenCV CPU vs OpenCV CUDA:" << std::endl;
                            ::compareImage(oOutputImageCPU(roi), oOutputImageCUDA(roi));  


                            std::cout << "ImgProcLab vs OpenCV CPU:" << std::endl;
                            ::compareImage(oOutputImage(roi), oOutputImageCPU(roi));  
                            
                            std::cout << "ImgProcLab vs OpenCV CUDA:" << std::endl;
                            ::compareImage(oOutputImage(roi), oOutputImageCUDA(roi));  


                            cv::imshow("GaussianFilter", oOutputImage);
                                    
                            cv::waitKey(0);
                            delete[] pOut;
                        }
                    }
                    break;
                }
                case Version::V2:
                {
                    std::cout << "Executing GaussianFilter V2" << std::endl;
                    std::cout << "Gaussian Kernel Launcher" << std::endl;

                    ::displayInfo(m_eConfig);
                    cv::Mat oImage = cv::imread(std::string(*m_eConfig.m_sImagePath));
                    ImgProcLab::Filter::GaussianFilter oGaussianFilter( oImage,
                                                                        *m_eConfig.m_iKernelSize,
                                                                        *m_eConfig.m_fSigma,
                                                                        ImgProcLab::GaussianVersion::V2);
                            
                    m_eGaussianData =  std::move(oGaussianFilter.getGaussianData());
                    
                    float fElapsedTime = executeCuda([&](){ ImgProcLab::detail::gaussianV2(m_eGaussianData);}, 
                                                     m_eConfig.m_bBenchmark, 
                                                     m_eConfig.m_iGPUWarmUps, 
                                                     m_eConfig.m_iIterations);

                    int iHeight   = m_eGaussianData.m_eCudaExecutionContext.m_iHeight;   
                    int iWidth    = m_eGaussianData.m_eCudaExecutionContext.m_iWidth;
                    assert(oImage.rows == iHeight);
                    assert(oImage.cols == iWidth);

                    cv::Mat oOutputImageCPU;
                    cv::Mat oOutputImageCUDA;
                    cv::Mat oOutputImage;                    
                    if(m_eConfig.m_bBenchmark)
                    {
                        const std::string& sPath = GetBenchmarkDir() + "gaussianV1.csv";
                        BenchmarResult oBenchmarResult{iWidth,iHeight,
                            *m_eConfig.m_iKernelSize,
                            m_eConfig.m_iGPUWarmUps,
                            m_eConfig.m_iIterations,
                            fElapsedTime,
                            "gaussian",
                            "v1"};  
                        
                        BenchmarkWriter::writeCSV(sPath, oBenchmarResult);

                        float fElapsedTImeCPU = executeOpenCV([&](){cv::GaussianBlur(oImage,
                                                                                     oOutputImageCPU, 
                                                                                     cv::Size(*m_eConfig.m_iKernelSize,*m_eConfig.m_iKernelSize), 
                                                                                     *m_eConfig.m_fSigma,
                                                                                     *m_eConfig.m_fSigma,
                                                                                     cv::BORDER_CONSTANT);}, 
                                                                    Backend::CPU, 
                                                                    m_eConfig.m_bBenchmark,
                                                                    m_eConfig.m_iGPUWarmUps,
                                                                    m_eConfig.m_iIterations);



                        BenchmarResult oBenchmarResultCPU{iWidth,
                                                          iHeight,    
                                                          *m_eConfig.m_iKernelSize,   
                                                          m_eConfig.m_iGPUWarmUps,
                                                          m_eConfig.m_iIterations,
                                                          fElapsedTImeCPU,
                                                          "gaussian",
                                                          "CPU"};
            
                        BenchmarkWriter::writeCSV(sPath, oBenchmarResultCPU);


                            
                            
                        cv::cuda::GpuMat d_input, d_output;
                        d_input.upload(oImage);
                        cv::cuda::Stream stream;
                        auto oGaussian = cv::cuda::createGaussianFilter(oImage.type(), 
                                                                        oImage.type(),
                                                                        cv::Size(*m_eConfig.m_iKernelSize, *m_eConfig.m_iKernelSize), 
                                                                        *m_eConfig.m_fSigma, 
                                                                        *m_eConfig.m_fSigma,
                                                                        cv::BORDER_CONSTANT);
                                                                        
                        float fElapsedTImeCUDA = executeOpenCV([&](){oGaussian->apply(d_input, d_output, stream);},
                                                                    Backend::CUDA, 
                                                                    m_eConfig.m_bBenchmark, 
                                                                    m_eConfig.m_iGPUWarmUps,
                                                                    m_eConfig.m_iIterations);

                        d_output.download(oOutputImageCUDA);
                        BenchmarResult oBenchmarResultCUDA{iWidth,
                            iHeight,
                            *m_eConfig.m_iKernelSize,
                            m_eConfig.m_iGPUWarmUps, 
                            m_eConfig.m_iIterations, 
                            fElapsedTImeCUDA,
                            "gaussian", 
                            "CUDA"};

                        BenchmarkWriter::writeCSV(sPath, oBenchmarResultCUDA);
                        
                        bool bVisualize = true;
                        if( bVisualize )
                        {
                            cv::imshow("Original", oImage);
                            cv::imshow("GaussianFilterCPU", oOutputImageCPU);
                            cv::imshow("GaussianFilterCUDA", oOutputImageCUDA);


                            constexpr int iChannels = 3;
                            uchar* pOut = new uchar[iWidth*iHeight*iChannels]; 
                                
                            m_eGaussianData.m_eCudaExecutionContext.m_oDeviceOutputs.copyFromDevice(pOut);
                            oOutputImage = cv::Mat(iHeight, iWidth, CV_8UC3, pOut);
                

                            int radius = *m_eConfig.m_iKernelSize / 2;

                            cv::Rect roi(
                                radius,
                                radius,
                                oOutputImage.cols - 2 * radius,
                                oOutputImage.rows - 2 * radius
                            );

                            std::cout << "OpenCV CPU vs OpenCV CUDA:" << std::endl;
                            ::compareImage(oOutputImageCPU(roi), oOutputImageCUDA(roi));  


                            std::cout << "ImgProcLab vs OpenCV CPU:" << std::endl;
                            ::compareImage(oOutputImage(roi), oOutputImageCPU(roi));  
                            
                            std::cout << "ImgProcLab vs OpenCV CUDA:" << std::endl;
                            ::compareImage(oOutputImage(roi), oOutputImageCUDA(roi));  


                            cv::imshow("GaussianFilter", oOutputImage);
                                    
                            cv::waitKey(0);
                            delete[] pOut;
                        }
                    }
                    break;
                }      
                default:
                {
                    throw std::invalid_argument("[ERROR]: Executing GaussianFilter with Unknown Version");
                }
                break;
            }
            break;
        }
        default:
        {
	        throw std::invalid_argument("Executing an Unknown Algorithm");
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
