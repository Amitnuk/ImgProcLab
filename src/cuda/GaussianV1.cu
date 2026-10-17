#include <opencv2/core/cuda.hpp>
#include "ImgProc/cuda/GaussianCuda.hpp"
#include "cuda/CudaUtils.hpp"

namespace ImgProcLab::detail
{
    __global__ void GaussianFilterKernel(const uchar* pIn, uchar* pOut, float* pKernel,  int iHeight, int iWidth, int iChannels = 3, const std::size_t iKernelSize = 3 )
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        
        if(col < iWidth && row < iHeight )
        {
            int half = iKernelSize/2;
            float vBGR[3] = {0.0f, 0.0f, 0.0f}; 
            for( int y = -half; y <= half; ++y )
            {
                for (int x = -half; x <= half; ++x)
                {
                    int iCol = col + x;
                    int iRow = row + y;
                    
                    if(( iCol < iWidth && iCol >= 0 ) && (iRow < iHeight && iRow >= 0))
                    {
                        int iImageIndex  = (iRow*iWidth + iCol)*iChannels;
                        int iKernelIndex = (y+half)*iKernelSize + (x + half);

                        vBGR[0] += static_cast<float>(pIn[ iImageIndex     ]) * pKernel[iKernelIndex] ;
                        vBGR[1] += static_cast<float>(pIn[ iImageIndex + 1 ]) * pKernel[iKernelIndex] ;
                        vBGR[2] += static_cast<float>(pIn[ iImageIndex + 2 ]) * pKernel[iKernelIndex] ;

                    }
                    
                }
                
                
            }

            int iIndex = (row*iWidth + col)*iChannels;
            pOut[ iIndex + 0 ] = static_cast<uchar>(vBGR[0]);
            pOut[ iIndex + 1 ] = static_cast<uchar>(vBGR[1]);
            pOut[ iIndex + 2 ] = static_cast<uchar>(vBGR[2]);

        }
    }
    void gaussianV1(const cv::Mat& InputImage,  
                    cv::Mat& OutputImage, 
                    const std::size_t& iKernelSize/*=3*/,  
                    const float& fSigma/*=1.0f*/, 
                    const std::size_t& iChannels/*=3*/)
    {
            auto [iWidth, iHeight]  = InputImage.size();    
            std::cout << "(height, width)="<< "("<< iHeight << "," << iWidth<<")"<< std::endl;
    }
}