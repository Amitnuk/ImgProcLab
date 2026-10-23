#include "ImgProc/cuda/gaussian/GaussianCuda.hpp"
#include "ImgProc/cuda/gaussian/GaussianData.hpp"
#include "common/ImgProcLabUtils.hpp"
#include "cuda/common/CudaExecutionContext.hpp"
#include <stdexcept>


namespace ImgProcLab
{
  namespace detail
  {
    extern __constant__ float GFILTER[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
    constexpr int CHANNELS = 3;
    __global__ void GaussianFilterKernel(const uchar* pIn, uchar* pOut,  int iHeight, int iWidth, const std::size_t iKernelSize = 3 )
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;


        if(col < iWidth && row < iHeight) 
        {
           float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};
           const int iHalf = iKernelSize/2;
           for(int y = -iHalf; y <=iHalf; ++y) 
           {
             for(int x = -iHalf; x <=iHalf; ++x) 
             {
               int iCol = col + x;
               int iRow = row + y;
             
               if( (iCol >= 0 && iCol <iWidth ) && (iRow >= 0 && iRow < iHeight) ) 
               {
                   int iImageIndex  = (iRow * iWidth + iCol)*CHANNELS;
                   int iKernelIndex = (y + iHalf)*iKernelSize + (x + iHalf);
                   const float& fWeight = GFILTER[iKernelIndex];

                   vBGR[0] += static_cast<float>(pIn[iImageIndex + 0]) * fWeight;
                   vBGR[1] += static_cast<float>(pIn[iImageIndex + 1]) * fWeight;
                   vBGR[2] += static_cast<float>(pIn[iImageIndex + 2]) * fWeight;
               }
             }
           }

           int iImageIndex  = (row * iWidth + col)*CHANNELS;
           pOut[iImageIndex + 0] = static_cast<uchar>(__float2int_rn(vBGR[0]));
           pOut[iImageIndex + 1] = static_cast<uchar>(__float2int_rn(vBGR[1]));
           pOut[iImageIndex + 2] = static_cast<uchar>(__float2int_rn(vBGR[2]));

        }
    }
    __global__ void GaussianFilterKernelHorizontal(const uchar* pIn, float* pOut,  int iHeight, int iWidth, const std::size_t iKernelSize = 3 )
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;


        if(col < iWidth && row < iHeight)
        {
            float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};
            int iHalf = iKernelSize/2;

            for(int x = -iHalf; x <= iHalf; ++x)
            {
                int iCol = col + x;
                if( iCol >=0 && iCol < iWidth) 
                {
                    
                    int iImageIndex  = (row * iWidth + iCol)*CHANNELS;
                    int iKernelIndex = x + iHalf;
                    const float& fWeights = GFILTER[iKernelIndex];
                    vBGR[0] += static_cast<float>(pIn[iImageIndex + 0]) * fWeights; 
                    vBGR[1] += static_cast<float>(pIn[iImageIndex + 1]) * fWeights;
                    vBGR[2] += static_cast<float>(pIn[iImageIndex + 2]) * fWeights; 

                }
                
            }
            int iImageIndex = (row *iWidth + col)*CHANNELS;
            pOut[iImageIndex + 0] = vBGR[0];
            pOut[iImageIndex + 1] = vBGR[1];
            pOut[iImageIndex + 2] = vBGR[2];            
        }

    }

    __global__ void GaussianFilterKernelVertical(const float*  pIn, uchar* pOut,  int iHeight, int iWidth, const std::size_t iKernelSize = 3 )
    {
       int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;


        if(col < iWidth && row < iHeight)
        {
            float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};
            int iHalf = iKernelSize/2;

            for(int y = -iHalf; y <= iHalf; ++y)
            {
                int iRow = row + y;
                if( iRow >=0 && iRow < iHeight) 
                {
                    
                    int iImageIndex  = (iRow * iWidth + col)*CHANNELS;
                    int iKernelIndex = y + iHalf;
                    const float& fWeights = GFILTER[iKernelIndex];
                    vBGR[0] += pIn[iImageIndex + 0] * fWeights; 
                    vBGR[1] += pIn[iImageIndex + 1] * fWeights;
                    vBGR[2] += pIn[iImageIndex + 2] * fWeights; 

                }
                
            }
            int iImageIndex = (row *iWidth + col)*CHANNELS;
            pOut[iImageIndex + 0] = static_cast<uchar>(__float2int_rn(vBGR[0]));
            pOut[iImageIndex + 1] = static_cast<uchar>(__float2int_rn(vBGR[1]));
            pOut[iImageIndex + 2] = static_cast<uchar>(__float2int_rn(vBGR[2]));
        }

    }

    void gaussianV3(GaussianData& eData, const int iThreads /*=DEFAULT_THREADS*/, const Separable eCase /*= Separable::FALSE*/)
    {

        CudaExecutionContext& eCudaContext = eData.m_eCudaExecutionContext;
      
        const int& iHeight  = eCudaContext.m_iHeight;   
        const int& iWidth   = eCudaContext.m_iWidth; 
        
        int iGridDimX = (iWidth + iThreads - 1)/iThreads;
        int iGridDimY = (iHeight + iThreads - 1)/iThreads;
        
        dim3 oGridDim(iGridDimX, iGridDimY);
        dim3 oBlockDim(iThreads, iThreads);
        
        if(eCase == Separable::FALSE)
        {
            GaussianFilterKernel<<<oGridDim,oBlockDim>>>(eCudaContext.m_oDeviceInput, eCudaContext.m_oDeviceOutputs, iHeight, iWidth, eData.m_iKernelSize);
        }
        else if(eCase == Separable::TRUE) 
        {
          GaussianFilterKernelHorizontal<<<oGridDim,oBlockDim>>>(eCudaContext.m_oDeviceInput, eCudaContext.m_oDeviceIntermediate, iHeight, iWidth, eData.m_iKernelSize);
          GaussianFilterKernelVertical<<<oGridDim,oBlockDim>>>(eCudaContext.m_oDeviceIntermediate, eCudaContext.m_oDeviceOutputs, iHeight, iWidth, eData.m_iKernelSize);
        }
        else 
        {
          throw std::invalid_argument("Unknown version of the Gaussian");
        }
    }  
  }
}
