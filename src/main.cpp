#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>



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
  cv::waitKey(0);  
    
  return 0;  
}  
