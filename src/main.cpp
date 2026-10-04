#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include "imgproc/filters/BoxFilter.hpp"
#include "imgproc/filters/GaussianFilter.hpp"
#include "imgproc/filters/SobelFilter.hpp"
#include <vector>


void BoxFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel  = 3)
{
  uchar* pIn  = mImage.reshape(1,1).ptr<uchar>(0);

  ImgProc::BoxFilter<uchar,uchar> oBoxFilter(iWidth, iHeight, iChannels, iThreads, iKernel);
  uchar* pOut = new uchar[iWidth*iHeight*iChannels]; 
  oBoxFilter.KernelLauncher(pIn, pOut);
  cv::Mat mAvgFilter(iHeight, iWidth, CV_8UC3, pOut);
  cv::imshow("Box Filter", mAvgFilter);

  delete[] pOut;
}

void GaussianFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel = 7, float fSigma = 0.5f )
{
  cv::Mat intImage;
  mImage.convertTo(intImage, CV_32SC3);
  
  int* pIn  = intImage.reshape(1,1).ptr<int>(0);
  ImgProc::GaussianFilter<int, int> oGaussianFilter(iWidth, iHeight, iChannels, iThreads, iKernel, iKernel);
  int* pOut = new int[iWidth*iHeight*iChannels];
  oGaussianFilter.KernelLauncher(pIn, pOut);
  cv::Mat displayImageGaussianFilter;
  cv::Mat mGaussianFilter(iHeight, iWidth, CV_32SC3, pOut);
  mGaussianFilter.convertTo(displayImageGaussianFilter, CV_8UC3);
  cv::imshow("Gaussian Filter", displayImageGaussianFilter);

  delete [] pOut;

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

  ImgProc::SobelFilter<int, int> oSobelFilter(iWidth, iHeight, iThreads, iKernel);
 
  int* pOut = new int[iWidth*iHeight];
  oSobelFilter.KernelLauncher(pIn, pOut);

  
  cv::Mat displayImageobelFilter;
  cv::Mat mSobelFilter(iHeight, iWidth, CV_32SC1, pOut);
  mSobelFilter.convertTo(displayImageobelFilter, CV_8UC1);
  cv::imshow("Sobel Filter", displayImageobelFilter);


  delete [] pOut;

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
  int iThreads = 16;
  int iKernelSize = 3;
  float fSigma = 0.25f;


  
  BoxFilter(mImage, iWidth, iHeight, iChannels, iThreads, iKernelSize );

  iKernelSize = 7;
  GaussianFilter(mImage, iWidth, iHeight, iChannels, iThreads, iKernelSize, fSigma);

  iKernelSize = 3;
  SobelFilter(mImage, iWidth, iHeight, iThreads, iKernelSize);

  

  
  cv::waitKey(0);  
  return 0;  
}  
