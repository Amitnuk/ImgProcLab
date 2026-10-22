
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include "AlgorithmRunner.hpp"
#include "BenchmarkWriter.hpp"
#include <opencv2/cudafilters.hpp> 
#include <ImgProc/cuda/gaussian/GaussianData.hpp>
#include <ImgProc/filters/GaussianFilter.hpp>
#include <cuda/common/CudaExecution.hpp>
#include <cuda/common/OpenCVExecution.hpp>
#include <ImgProc/cuda/gaussian/GaussianCuda.hpp>
#include <common/ImgProcLabUtils.hpp>
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
                            
                    m_eGaussianData = oGaussianFilter.getGaussianData();

                    //TODO: 
                    // oGaussianFilter..getGaussianData() performs a move so oGaussianFilter does not have it anymore, should allow copy, Array should be copiable

                    bool bVisualize = true;
                    if(m_eConfig.m_bBenchmark)
                    {
                        benchmark(ImgProcLab::detail::gaussianV1,
                                  oImage,
                                  "gaussian",
                                  "V1",
                                  true);
                    }
                    else if(bVisualize)
                    {

                      executeCuda([&](){ImgProcLab::detail::gaussianV1(m_eGaussianData, DEFAULT_THREADS);},false, 0, 0);

                      constexpr int iChannels = 3;
                      const auto [iWidth, iHeight] = oImage.size();
                      cv::Mat oOutputImage = cv::Mat(iHeight, iWidth, CV_8UC3);
                      
                      m_eGaussianData.m_eCudaExecutionContext.m_oDeviceOutputs.copyFromDevice(oOutputImage.data);
                      
                      cv::imshow("Original Image", oImage);
                      cv::imshow("Gaussian Blurred Image", oOutputImage);
                      cv::waitKey(0);
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
                    
                    
                    bool bVisualize = true;
                    if(m_eConfig.m_bBenchmark)
                    {
                        benchmark(ImgProcLab::detail::gaussianV2,
                                  oImage,
                                  "gaussian",
                                  "V2",
                                  bVisualize);
                    }
                    else if(bVisualize)
                    {

                      executeCuda([&](){ImgProcLab::detail::gaussianV2(m_eGaussianData, DEFAULT_THREADS);},false, 0, 0);
                      constexpr int iChannels = 3;
                      const auto [iWidth, iHeight] = oImage.size();
                      uchar* pOut = new uchar[iWidth * iHeight * iChannels];
                      
                      m_eGaussianData.m_eCudaExecutionContext.m_oDeviceOutputs.copyFromDevice(pOut);
                      cv::Mat oOutputImage = cv::Mat(iHeight, iWidth, CV_8UC3, pOut); 

                      cv::imshow("Original Image", oImage);
                      cv::imshow("Gaussian Blurred Image", oOutputImage);
                      cv::waitKey(0);

                      
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

//TODO: make this function filter/algorithms agnostic, right now only works for Gaussian Blur 
template<typename Function>
void AlgorithmRunner::benchmark(Function&& function,
                                const cv::Mat& oImage,
                                const std::string& sAlgorithm,
                                const std::string& sVersion,
                                bool bVisualize /*= true*/)
{ 


  float fElapsedTime = executeCuda([&](){ function(m_eGaussianData, DEFAULT_THREADS);}, // ;
                                     m_eConfig.m_bBenchmark, 
                                     m_eConfig.m_iGPUWarmUps, 
                                     m_eConfig.m_iIterations);

    
    assert(oImage.rows == m_eGaussianData.m_eCudaExecutionContext.m_iHeight);
    assert(oImage.cols == m_eGaussianData.m_eCudaExecutionContext.m_iWidth);

    auto [iWidth, iHeight] = oImage.size();
    const std::string& sPath = GetBenchmarkDir() + sAlgorithm + sVersion;
    BenchmarResult oBenchmarResult{iWidth,iHeight,
                                   *m_eConfig.m_iKernelSize,
                                   m_eConfig.m_iGPUWarmUps,
                                   m_eConfig.m_iIterations,
                                   fElapsedTime,
                                   sAlgorithm,
                                   sVersion};  
                        
    BenchmarkWriter::writeCSV(sPath, oBenchmarResult);

    cv::Mat oOutputImageCPU;
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
                                      sAlgorithm,
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

    cv::Mat oOutputImageCUDA;
    d_output.download(oOutputImageCUDA);
    BenchmarResult oBenchmarResultCUDA{iWidth,
                                       iHeight,
                                       *m_eConfig. m_iKernelSize,
                                       m_eConfig.m_iGPUWarmUps, 
                                       m_eConfig.m_iIterations, 
                                       fElapsedTImeCUDA,
                                       sAlgorithm, 
                                       "CUDA"};

    BenchmarkWriter::writeCSV(sPath, oBenchmarResultCUDA);
                        
    
    if( bVisualize )
    {
      cv::imshow("Original", oImage);
      cv::imshow("GaussianFilterCPU", oOutputImageCPU);
      cv::imshow("GaussianFilterCUDA", oOutputImageCUDA);
      

      constexpr int iChannels = 3;

      cv::Mat oOutputImage = cv::Mat(iHeight, iWidth, CV_8UC3);
      m_eGaussianData.m_eCudaExecutionContext.m_oDeviceOutputs.copyFromDevice(oOutputImage.data);
      
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


      cv::imshow("GuassianFilter", oOutputImage);
      
      cv::waitKey(0);
     
    }

}

void AlgorithmRunner::run()
{
    prepare();

}
