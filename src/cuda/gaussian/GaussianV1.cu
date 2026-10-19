#include <opencv2/core/cuda.hpp>
#include "ImgProc/cuda/gaussian/GaussianData.hpp"
#include "cuda/common/CudaUtils.hpp"

/*

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

                        vBGR[0] += static_cast<float>(pIn[ iImageIndex     ])  ;
                        vBGR[1] += static_cast<float>(pIn[ iImageIndex + 1 ]) ;
                        vBGR[2] += static_cast<float>(pIn[ iImageIndex + 2 ])  ;

                    }
                    
                }
                
                
            }

            int iIndex = (row*iWidth + col)*CHANNELS;
            pOut[ iIndex + 0 ] = static_cast<uchar>(vBGR[0]);
            pOut[ iIndex + 1 ] = static_cast<uchar>(vBGR[1]);
            pOut[ iIndex + 2 ] = static_cast<uchar>(vBGR[2]);

        }


*/

namespace ImgProcLab::detail
{
    constexpr int CHANNELS = 3;
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
                    
                    if((iCol < iWidth && iCol >= 0) && (iRow < iHeight && iRow >= 0))
                    {
                        int iImageIndex  = (iRow*iWidth + iCol)*CHANNELS;
                        int iKernelIndex = (y + half)*iKernelSize + (x + half);

                        vBGR[0] += static_cast<float>(pIn[iImageIndex + 0]) * pKernel[iKernelIndex];
                        vBGR[1] += static_cast<float>(pIn[iImageIndex + 1]) * pKernel[iKernelIndex];
                        vBGR[2] += static_cast<float>(pIn[iImageIndex + 2]) * pKernel[iKernelIndex];

                    }
                    
                }
            }

            int iIndex = (row*iWidth + col)*CHANNELS;
            pOut[ iIndex + 0 ] = static_cast<uchar>(vBGR[0]);
            pOut[ iIndex + 1 ] = static_cast<uchar>(vBGR[1]);
            pOut[ iIndex + 2 ] = static_cast<uchar>(vBGR[2]);
        }
        
    }


    
    void gaussianV1(GaussianData& eData)
    {

        CudaExecutionContext& eCudaExContex = eData.m_eCudaExecutionContext;
        
        int iHeight  = eCudaExContex.m_iHeight;   
        int iWidth   = eCudaExContex.m_iWidth; 

        
        std::cout << "Gaussian Kernel Launcher " << std::endl;
        int iThreads = 32;
        int iGridDimX = (iWidth  + iThreads - 1)/iThreads;
        int iGridDimY = (iHeight + iThreads - 1)/iThreads;

        dim3 oGridDim(iGridDimX, iGridDimY);
        dim3 oBlockDim(iThreads, iThreads); 

        std::cout << "grid=="<< "("<< iGridDimX << "," <<iGridDimY<<")"<< std::endl;
        std::cout << "block=="<< "("<< iThreads << "," <<iThreads<<")"<< std::endl;

        cudaEvent_t oStart, oStop;
        float fTimeInMS;

    
        CUDA_CHECK(cudaEventCreate(&oStart));
        CUDA_CHECK(cudaEventCreate(&oStop));
        CUDA_CHECK(cudaEventRecord(oStart));
        

        GaussianFilterKernel<<<oGridDim,oBlockDim>>>(eCudaExContex.m_oDeviceInput.Data(),
                                                     eCudaExContex.m_oDeviceOutputs.Data(),
                                                     eCudaExContex.m_oKernel.Data(),
                                                     iHeight,
                                                     iWidth,
                                                     eData.m_iKernelSize);

        CUDA_CHECK(cudaEventRecord(oStop));
        CUDA_CHECK(cudaEventSynchronize(oStop));
        
        CUDA_CHECK(cudaEventElapsedTime(&fTimeInMS, oStart, oStop));
        std::cout << "CUDA kernel: " << fTimeInMS << "ms\n";

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());


    }

    
}