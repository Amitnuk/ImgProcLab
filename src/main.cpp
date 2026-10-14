#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp> 
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "ImgProc/filters/BoxFilter.hpp"
#include "ImgProc/filters/MedianFilter.hpp"
#include "ImgProc/filters/GaussianFilter.hpp"
#include "ImgProc/EdgeDetection/SobelFilter.hpp"
#include "ImgProc/Thresholding/BinaryThreshold.hpp"
#include "ImgProc/colorConversion/BGR2Gray.hpp"

#include <vector>

using namespace ImgProc;

void ColorConversion(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads);
void BoxFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel);
void MedianFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel);
void GaussianFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel , float fSigma );
void SobelFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iThreads, int iKernel);
void BinaryThreshold(const cv::Mat& mImage, int iWidth, int iHeight, int iThreads, int iKernel, int iThreshold);


void GaussianFilter(cv::Mat oImage, int ikernelSize = 3, double fSigma = 0.1)
{
    
    cv::cuda::GpuMat d_input, d_output;
    d_input.upload(oImage);

   

    auto gaussian = cv::cuda::createGaussianFilter(oImage.type(), oImage.type(),cv::Size(ikernelSize, ikernelSize), fSigma);

    cv::cuda::Stream stream;

    // Warm-up: avoid including initialization overhead.
    for (int i = 0; i < 1; ++i)
        gaussian->apply(d_input, d_output, stream);

    stream.waitForCompletion();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Use the same CUDA stream as OpenCV.
    cudaStream_t cudaStream =
        cv::cuda::StreamAccessor::getStream(stream);

    cudaEventRecord(start, cudaStream);

    constexpr int iterations = 1;

    for (int i = 0; i < iterations; ++i)
        gaussian->apply(d_input, d_output, stream);

    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);

    std::cout << "OpenCV CUDA Gaussian filter: " << elapsedMs / iterations << " ms per iteration\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void GaussianFilterCPU(cv::Mat oImage, int ikernelSize = 3, double fSigma = 0.1)
{
    
    cv::Mat output;

    const int kernelSize = 5;
    const double sigma = 1.0;

    // Warm-up
    for (int i = 0; i < 10; ++i)
    {
        cv::GaussianBlur(
            oImage,
            output,
            cv::Size(kernelSize, kernelSize),
            sigma,
            sigma,
            cv::BORDER_CONSTANT
        );
    }

    constexpr int iterations = 100;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        cv::GaussianBlur(
            oImage,
            output,
            cv::Size(kernelSize, kernelSize),
            sigma,
            sigma,
            cv::BORDER_CONSTANT
        );
    }

    auto stop = std::chrono::steady_clock::now();

    double elapsedMs =
        std::chrono::duration<double, std::milli>(
            stop - start
        ).count();

    std::cout << "OpenCV CPU Gaussian filter: "
              << elapsedMs / iterations
              << " ms per iteration\n";

    
}


int main(int argc, char* argv[]) {

  
  if( argc != 2 ){
    std::cout << "Introduce Image ./exec filename" << std::endl;
    return 1;
  }
  
  std::cout << "args = "<< argv[0] << " " << argv[1] << std::endl;

  cv::Mat mImage = cv::imread(argv[1]);
  
  if( mImage.empty() ){
    std::cout << "Failed to load, check that " << argv[1] << " does exits" << std::endl;
  }

  cv::imshow("Original", mImage);
  int iWidth  = mImage.cols;
  int iHeight = mImage.rows;
  int iChannels = 3;
  int iThreads = 32;
  int iKernelSize = 3;
  float fSigma = 1.5f;
  int iThreshold = 15;


  //ColorConversion(mImage, iWidth, iHeight, iChannels, iThreads);
  //BoxFilter(mImage, iWidth, iHeight, iChannels, iThreads, iKernelSize );

  
  iKernelSize = 5;
  GaussianFilter(mImage, iWidth, iHeight, iChannels, iThreads, iKernelSize, fSigma);

  //iKernelSize = 3;
  //SobelFilter(mImage, iWidth, iHeight, iThreads, iKernelSize);

  //BinaryThreshold(mImage, iWidth, iHeight, iThreads, iKernelSize, iThreshold);
  
  //iKernelSize = 3;
  //iThreads = 32;
  //MedianFilter(mImage, iWidth, iHeight, iChannels, iThreads, iKernelSize );
  
  cv::waitKey(0);
  
  return 0;  
}  


void ColorConversion(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads)
{
  uchar* pIn  = mImage.reshape(1,1).ptr<uchar>(0);

  color::BGR2Gray<uchar,uchar> oBGR2Gray(iWidth, iHeight, iChannels, iThreads);

  uchar* pOut = new uchar[iWidth*iHeight]; 
  oBGR2Gray.KernelLauncher(pIn, pOut);
  cv::Mat mGray(iHeight, iWidth, CV_8UC1, pOut);
  cv::imshow("Color Conversion", mGray);

  delete[] pOut;
}


void BoxFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel  = 3)
{
  uchar* pIn  = mImage.reshape(1,1).ptr<uchar>(0);

  Filter::BoxFilter<uchar,uchar> oBoxFilter(iWidth, iHeight, iChannels, iThreads, iKernel);
  uchar* pOut = new uchar[iWidth*iHeight*iChannels]; 
  oBoxFilter.KernelLauncher(pIn, pOut);
  cv::Mat mAvgFilter(iHeight, iWidth, CV_8UC3, pOut);
  cv::imshow("Box Filter", mAvgFilter);

  delete[] pOut;
}

void MedianFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel  = 3)
{
  uchar* pIn  = mImage.reshape(1,1).ptr<uchar>(0);

  Filter::MedianFilter<uchar,uchar> oMedianFilter(iWidth, iHeight, iChannels, iThreads, iKernel);
  uchar* pOut = new uchar[iWidth*iHeight*iChannels]; 
  oMedianFilter.KernelLauncher(pIn, pOut);
  cv::Mat mAvgFilter(iHeight, iWidth, CV_8UC3, pOut);
  cv::imshow("Median Filter", mAvgFilter);

  delete[] pOut;
}

void GaussianFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel = 7, float fSigma = 0.5f )
{
  cv::Mat intImage;
  mImage.convertTo(intImage, CV_32SC3);
  
  int* pIn  = intImage.reshape(1,1).ptr<int>(0);
  Filter::GaussianFilter<int, int> oGaussianFilter(iWidth, iHeight, iChannels, iThreads, iKernel, iKernel);
  int* pOut = new int[iWidth*iHeight*iChannels];
  oGaussianFilter.KernelLauncher(pIn, pOut);
  cv::Mat displayImageGaussianFilter;
  cv::Mat mGaussianFilter(iHeight, iWidth, CV_32SC3, pOut);
  mGaussianFilter.convertTo(displayImageGaussianFilter, CV_8UC3);
  cv::imshow("Gaussian Filter", displayImageGaussianFilter);

  delete [] pOut;

  GaussianFilter(mImage, iKernel);
  GaussianFilterCPU(mImage, iKernel);

}


void SobelFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iThreads, int iKernel = 3)
{
  cv::Mat mImgGray;
  cv::cvtColor(mImage, mImgGray, cv::COLOR_BGR2GRAY);

  /*
    cv::imshow("Gray", mImgGray);
  */
   
  cv::Mat intImage;
  mImgGray.convertTo(intImage, CV_32SC1);
  
  int* pIn  = intImage.reshape(1,1).ptr<int>(0);

  /* 
  cv::Mat intImageDisplay;
  intImage.convertTo(intImageDisplay, CV_8UC1);
  cv::imshow("INIT", intImageDisplay);
  */

  ImgProc::Edge::SobelFilter <int, int> oSobelFilter(iWidth, iHeight, iThreads, iKernel);
 
  int* pOut = new int[iWidth*iHeight];
  oSobelFilter.KernelLauncher(pIn, pOut);

  
  cv::Mat displayImageobelFilter;
  cv::Mat mSobelFilter(iHeight, iWidth, CV_32SC1, pOut);
  mSobelFilter.convertTo(displayImageobelFilter, CV_8UC1);
  cv::imshow("Sobel Filter", displayImageobelFilter);


  delete [] pOut;

}


void BinaryThreshold(const cv::Mat& mImage, int iWidth, int iHeight, int iThreads, int iKernel = 3, int iThreshold = 150)
{
  cv::Mat mImgGray;
  cv::cvtColor(mImage, mImgGray, cv::COLOR_BGR2GRAY);

  /*
    cv::imshow("Gray", mImgGray);
  */
   
  cv::Mat intImage;
  mImgGray.convertTo(intImage, CV_32SC1);
  
  int* pIn  = intImage.reshape(1,1).ptr<int>(0);

  /* 
  cv::Mat intImageDisplay;
  intImage.convertTo(intImageDisplay, CV_8UC1);
  cv::imshow("INIT", intImageDisplay);
  */

  ImgProc::Edge::SobelFilter<int, int> oSobelFilter(iWidth, iHeight, iThreads, iKernel);
 
  int* pOut = new int[iWidth*iHeight];
  oSobelFilter.KernelLauncher(pIn, pOut);

  
  
  cv::Mat mSobelFilter(iHeight, iWidth, CV_32SC1, pOut);
  pIn  = mSobelFilter.reshape(1,1).ptr<int>(0);
  Thresh::BinaryThreshold<int, int> BinaryThreshold(iWidth, iHeight, iThreads, iThreshold);
  BinaryThreshold.KernelLauncher(pIn, pOut);

  cv::Mat mBinaryThesholdFilter(iHeight, iWidth, CV_32SC1, pOut);

  cv::Mat displayBinaryThresoldFilter;
  mBinaryThesholdFilter.convertTo(displayBinaryThresoldFilter, CV_8UC1);
  cv::imshow("Binary Threshold Filter", displayBinaryThresoldFilter);

  cv::Mat display;
  cv::cvtColor( displayBinaryThresoldFilter, display,cv::COLOR_GRAY2BGR);


  cv::Mat displayCartoon ;
  cv::subtract(mImage, display, displayCartoon);
  cv::imshow("Cartoonize", displayCartoon);
  delete [] pOut;

}
