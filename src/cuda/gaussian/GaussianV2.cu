#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include "ImgProc/cuda/gaussian/GaussianCuda.hpp"
#include "ImgProc/cuda/gaussian/GaussianData.hpp"
#include "common/GaussianUtils.hpp"
#include "cuda/common/CudaBuffer.hpp"
#include "cuda/common/CudaUtils.hpp"




namespace ImgProcLab::detail
{

  constexpr int CHANNELS = 3;
  __global__ void GaussianFilterKernelHorizontal(const uchar* pIn, float* pOut, const float* pKernel,  int iHeight, int iWidth, const std::size_t iKernelSize = 3 )
  {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;


        if( col < iWidth && row < iHeight)
        {
            int iHalf = iKernelSize/2;
            float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};

            for(int x = -iHalf; x <= iHalf; ++x)
            {
                int iCol = col + x;
                if(iCol >= 0 && iCol < iWidth )
                {
                    int iKernelIndex = x + iHalf;
                    int iImageIndex = (row*iWidth + iCol)*CHANNELS;
                    const float& fWeights = pKernel[iKernelIndex]; 
                    vBGR[0] += static_cast<float>(pIn[iImageIndex + 0]) * fWeights;
                    vBGR[1] += static_cast<float>(pIn[iImageIndex + 1]) * fWeights;
                    vBGR[2] += static_cast<float>(pIn[iImageIndex + 2]) * fWeights;
                  
                }
                

            }
            
            int iImageIndex = (row*iWidth + col)*CHANNELS;

            pOut[iImageIndex + 0] = vBGR[0];
            pOut[iImageIndex + 1] = vBGR[1];
            pOut[iImageIndex + 2] = vBGR[2];
        }
  }

  __global__ void GaussianFilterKernelVertical(const float* pIn, uchar* pOut, const float* pKernel,  int iHeight, int iWidth, const std::size_t iKernelSize = 3 )
  {
      int col = threadIdx.x + blockIdx.x * blockDim.x ;
      int row = threadIdx.y + blockIdx.y * blockDim.y ;


      if(col < iWidth && row < iHeight)
      {
          int iHalf = iKernelSize/2;
          float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};

          for (int y = -iHalf; y <= iHalf; ++y)
          {
              int iRow = y + row;
              if(iRow >= 0  && iRow < iHeight)
              {
                  int iKernelIndex = y + iHalf;
                  int iInmageIndex = (iRow*iWidth + col)*CHANNELS;
                  const float& fWeights = pKernel[iKernelIndex]; 
                  vBGR[0] += pIn[iInmageIndex + 0] * fWeights;
                  vBGR[1] += pIn[iInmageIndex + 1] * fWeights;
                  vBGR[2] += pIn[iInmageIndex + 2] * fWeights;
              }
              
          }

          int iImageIndex = (row*iWidth + col)*CHANNELS;

          pOut[iImageIndex + 0] = static_cast<uchar>(__float2int_rn(vBGR[0]));
          pOut[iImageIndex + 1] = static_cast<uchar>(__float2int_rn(vBGR[1]));
          pOut[iImageIndex + 2] = static_cast<uchar>(__float2int_rn(vBGR[2]));
      }
      
  }

  void gaussianV2(GaussianData& eData, const int iThreads/*=32*/, Separable )
  {

    CudaExecutionContext& eCudaContext = eData.m_eCudaExecutionContext;
        
    const int& iHeight  = eCudaContext.m_iHeight;   
    const int& iWidth   = eCudaContext.m_iWidth; 

    const int iGridDimX = (iWidth  + iThreads - 1)/iThreads;
    const int iGridDimY = (iHeight + iThreads - 1)/iThreads;

    dim3 oGridDim(iGridDimX, iGridDimY);
    dim3 oBlockDim(iThreads, iThreads); 


   
    GaussianFilterKernelHorizontal<<<oGridDim,oBlockDim>>>(eCudaContext.m_oDeviceInput.Data(),
                                                           eCudaContext.m_oDeviceIntermediate.Data(),
                                                           eCudaContext.m_oKernel.Data(),
                                                           iHeight,
                                                           iWidth,
                                                           eData.m_iKernelSize);

    
    GaussianFilterKernelVertical<<<oGridDim,oBlockDim>>>(eCudaContext.m_oDeviceIntermediate.Data(),
                                                         eCudaContext.m_oDeviceOutputs.Data(),
                                                         eCudaContext.m_oKernel.Data(),
                                                         iHeight,
                                                         iWidth,
                                                         eData.m_iKernelSize);
    
  }

}




