#include <opencv2/core/cuda.hpp>
#include "ImgProc/cuda/GaussianCuda.hpp"
#include "cuda/CudaUtils.hpp"
#include "cuda/CudaBuffer.hpp"

constexpr int CHANNELS = 3;

namespace ImgProcLab::detail
{
    __global__ void GaussianFilterKernel(const uchar* pIn, uchar* pOut, float* pKernel,  int iHeight, int iWidth, const std::size_t iKernelSize = 3 )
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
                        int iImageIndex  = (iRow*iWidth + iCol)*CHANNELS;
                        int iKernelIndex = (y+half)*iKernelSize + (x + half);

                        vBGR[0] += static_cast<float>(pIn[ iImageIndex     ]) * pKernel[iKernelIndex] ;
                        vBGR[1] += static_cast<float>(pIn[ iImageIndex + 1 ]) * pKernel[iKernelIndex] ;
                        vBGR[2] += static_cast<float>(pIn[ iImageIndex + 2 ]) * pKernel[iKernelIndex] ;

                    }
                    
                }
                
                
            }

            int iIndex = (row*iWidth + col)*CHANNELS;
            pOut[ iIndex + 0 ] = static_cast<uchar>(vBGR[0]);
            pOut[ iIndex + 1 ] = static_cast<uchar>(vBGR[1]);
            pOut[ iIndex + 2 ] = static_cast<uchar>(vBGR[2]);

        }
    }


    
    void gaussianV1(const cv::Mat& oInputImage,  
                    cv::Mat& oOutputImage, 
                    const std::size_t& iKernelSize/*=3*/,  
                    const float& fSigma/*=1.0f*/, 
                    const std::size_t& iChannels/*=3*/)
    {
            auto [iWidth, iHeight]  = oInputImage.size();    
            std::cout << "(height, width)="<< "("<< iHeight << "," << iWidth<<")"<< std::endl;

            std::cout << "Gaussian Kernel Launcher " << std::endl;
            int iThreads = 32;
            int iGridDimX = (iWidth  + iThreads - 1)/iThreads;
            int iGridDimY = (iHeight + iThreads - 1)/iThreads;

            dim3 oGridDim(iGridDimX, iGridDimY);
            dim3 oBlockDim(iThreads, iThreads); 

            int iImageSizeBytes = iWidth * iHeight * CHANNELS * sizeof(uchar);
            CudaBuffer<uchar> oCudaBufferIn(iImageSizeBytes);
            CudaBuffer<uchar> oCudaBufferOut(iImageSizeBytes);

           
            

            uchar* pIn_h  = oInputImage.reshape(1,1).ptr<uchar>(0);
            oCudaBufferIn.copyFromHost(pIn_h);

            

            cudaEvent_t oStart, oStop;
            float fTimeInMS;
            Array<float> aKernel = create2DGaussianKernel(iKernelSize, fSigma);

            float* pFloat = aKernel.data();
            int iKernelSizeBytes = iKernelSize*iKernelSize*sizeof(float);
            CudaBuffer<float> oCudaBufferKernel(iKernelSizeBytes);
            oCudaBufferKernel.copyFromHost(pFloat);

            CUDA_CHECK(cudaEventCreate(&oStart));
            CUDA_CHECK(cudaEventCreate(&oStop));
            CUDA_CHECK(cudaEventRecord(oStart));

            GaussianFilterKernel<<<oGridDim,oBlockDim>>>(oCudaBufferIn.Data(),
                                                         oCudaBufferOut.Data(),
                                                         oCudaBufferKernel.Data(),
                                                         iHeight,
                                                         iWidth,
                                                         iKernelSize);

            CUDA_CHECK(cudaEventRecord(oStop));
            CUDA_CHECK(cudaEventSynchronize(oStop));
            
            CUDA_CHECK(cudaEventElapsedTime(&fTimeInMS, oStart, oStop));
            std::cout << "CUDA kernel: " << fTimeInMS << "ms\n";

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            oOutputImage.create(iHeight, iWidth, CV_8UC3);
            oCudaBufferOut.copyFromDevice(oOutputImage.ptr<uchar>());

    }

    Array<float> create2DGaussianKernel(int iKernelSize, float fSigma)
    {   
        
        Array<float> aKernel(iKernelSize, iKernelSize);
        const float fInvTwoSigmaSquared = -0.5f / (fSigma * fSigma);
        int iHalf = iKernelSize/2;
        float sum = 0.0f;
        for (int y = -iHalf; y <= iHalf; ++y)
        {
            for (int x = -iHalf; x <= iHalf; ++x)
            {

                float fValue = (std::exp((x*x + y*y)*fInvTwoSigmaSquared));
                
                aKernel(y+iHalf, x+iHalf) = fValue;
                
                sum += fValue;
            }
        }

        const float invSum = 1.0f / sum;
        for (int y = 0; y < iKernelSize; ++y)
        {
            for( int x = 0; x < iKernelSize; ++x )
            {
                aKernel(y, x) *= invSum;
            }
        }
     
        return aKernel;
    }
}