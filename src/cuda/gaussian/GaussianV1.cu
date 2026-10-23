#include "ImgProc/cuda/gaussian/GaussianData.hpp"
#include "common/GaussianUtils.hpp"
#include "cuda/common/CudaUtils.hpp"

namespace ImgProcLab
{
    namespace detail 
    {
        constexpr int CHANNELS = 3;
  
      __global__ void GaussianFilterKernel(const uchar* pIn, uchar* pOut, const float* pKernel,  int iHeight, int iWidth, const std::size_t iKernelSize = 3 )
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
           
                int iImageIndex = (row*iWidth + col)*CHANNELS;
                pOut[ iImageIndex + 0 ] = static_cast<uchar>(__float2int_rn(vBGR[0]));
                pOut[ iImageIndex + 1 ] = static_cast<uchar>(__float2int_rn(vBGR[1]));
                pOut[ iImageIndex + 2 ] = static_cast<uchar>(__float2int_rn(vBGR[2]));
              }
        
      }


    
      void gaussianV1(GaussianData& eData, const int iThreads/*=32*/, Separable )
      {

            CudaExecutionContext& eCudaContext = eData.m_eCudaExecutionContext;
        
            const int& iHeight  = eCudaContext.m_iHeight;   
            const int& iWidth   = eCudaContext.m_iWidth; 

            const int iGridDimX = (iWidth  + iThreads - 1)/iThreads;
            const int iGridDimY = (iHeight + iThreads - 1)/iThreads;

            dim3 oGridDim(iGridDimX, iGridDimY);
            dim3 oBlockDim(iThreads, iThreads); 

        
            GaussianFilterKernel<<<oGridDim,oBlockDim>>>(eCudaContext.m_oDeviceInput.Data(),
                                                         eCudaContext.m_oDeviceOutputs.Data(),
                                                         eCudaContext.m_oKernel.Data(),
                                                         iHeight,
                                                         iWidth,
                                                         eData.m_iKernelSize);

      }

    }    
}
