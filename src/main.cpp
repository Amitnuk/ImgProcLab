#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include "imgproc/filters/blur/BoxFilter.hpp"
#include "imgproc/filters/blur/MedianFilter.hpp"
#include "imgproc/filters/blur/GaussianFilter.hpp"
#include "imgproc/filters/edge/SobelFilter.hpp"
#include "imgproc/filters/segmentation/BinaryThreshold.hpp"
#include <vector>

using namespace ImgProc;

void BoxFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel);
void MedianFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel);
void GaussianFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel , float fSigma );
void SobelFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iThreads, int iKernel);
void BinaryThreshold(const cv::Mat& mImage, int iWidth, int iHeight, int iThreads, int iKernel, int iThreshold);

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
  int iThreshold = 15;


  
  BoxFilter(mImage, iWidth, iHeight, iChannels, iThreads, iKernelSize );

  
  iKernelSize = 7;
  GaussianFilter(mImage, iWidth, iHeight, iChannels, iThreads, iKernelSize, fSigma);

  iKernelSize = 3;
  SobelFilter(mImage, iWidth, iHeight, iThreads, iKernelSize);

  BinaryThreshold(mImage, iWidth, iHeight, iThreads, iKernelSize, iThreshold);
  
  iKernelSize = 3;
  MedianFilter(mImage, iWidth, iHeight, iChannels, iThreads, iKernelSize );

  cv::waitKey(0);  
  return 0;  
}  





void BoxFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel  = 3)
{
  uchar* pIn  = mImage.reshape(1,1).ptr<uchar>(0);

  Blur::BoxFilter<uchar,uchar> oBoxFilter(iWidth, iHeight, iChannels, iThreads, iKernel);
  uchar* pOut = new uchar[iWidth*iHeight*iChannels]; 
  oBoxFilter.KernelLauncher(pIn, pOut);
  cv::Mat mAvgFilter(iHeight, iWidth, CV_8UC3, pOut);
  cv::imshow("Box Filter", mAvgFilter);

  delete[] pOut;
}

void MedianFilter(const cv::Mat& mImage, int iWidth, int iHeight, int iChannels, int iThreads, int iKernel  = 3)
{
  uchar* pIn  = mImage.reshape(1,1).ptr<uchar>(0);

  Blur::MedianFilter<uchar,uchar> oMedianFilter(iWidth, iHeight, iChannels, iThreads, iKernel);
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
  Blur::GaussianFilter<int, int> oGaussianFilter(iWidth, iHeight, iChannels, iThreads, iKernel, iKernel);
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
  Seg::BinaryThreshold<int, int> BinaryThreshold(iWidth, iHeight, iThreads, iThreshold);
  BinaryThreshold.KernelLauncher(pIn, pOut);

  cv::Mat mBinaryThesholdFilter(iHeight, iWidth, CV_32SC1, pOut);

  cv::Mat displayBinaryThresoldFilter;
  mBinaryThesholdFilter.convertTo(displayBinaryThresoldFilter, CV_8UC1);
  cv::imshow("Binary Threshold Filter", displayBinaryThresoldFilter);

  cv::Mat display;
  cv::cvtColor( displayBinaryThresoldFilter, display,cv::COLOR_GRAY2BGR);


  cv::Mat displayCartoon ;
  cv::subtract(mImage, display, displayCartoon);
  cv::imshow("Binary Threshold Filter 3", displayCartoon);
  delete [] pOut;

}