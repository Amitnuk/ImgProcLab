#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include "Filters.cuh"


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


  
  int N = 16;
  int iWidth  = mImage.cols;
  int iHeight = mImage.rows;
  Filters::AverageFilter oAvgFilter(iWidth, iHeight, 3, 16, 3);
  uchar* pIn  = mImage.reshape(1,1).ptr<uchar>(0);
  uchar* pOut = new uchar[iWidth*iHeight*3];

  oAvgFilter.KernelLauncher(pIn, pOut );
  
  cv::Mat mImageFiltered(iHeight, iWidth, CV_8UC3, pOut);
  
  //cv::Mat blur(iHeight, iWidth, CV_8UC3, out_h);

  cv::imshow("Filtered", mImageFiltered);
  cv::waitKey(0);  
  delete[] pOut;
  return 0;  
}  
