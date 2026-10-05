#include <iostream>
#include <stdio.h>
#include "imgproc/filters/edge/SobelFilter.hpp"
#include "cuda/CudaBuffer.hpp"



using uchar = unsigned char;


template<typename T, typename U, typename V>
__global__ void SobelFilterKernel(const T* pIn, U* pOut, V* pKernelX,  V* pKernelY, int iHeight, int iWidth, std::size_t iKernelSize=3, std::size_t fKernelFactor = 8.0f)
{
  int Col = threadIdx.x + blockIdx.x * blockDim.x ;
  int Row = threadIdx.y + blockIdx.y * blockDim.y ;
  
  if(Col < iWidth && Row < iHeight)
  {
    int iKernel = iKernelSize/2; 
    int iPixelValueX = 0;
    int iPixelValueY = 0;
    float fPixelValue = 0.0f;

    for(int u = -iKernel; u <= iKernel; ++u)
    {
      for (int v = -iKernel; v <= iKernel; ++v)
      {
        int iRow = Row + u;
        int iCol = Col + v;

        if( (iRow >= 0 && iRow < iHeight) && (iCol >= 0 && iCol < iWidth ))
        {
          int iPixelIndex  = iRow*iWidth + iCol;
          int iKernelIndex = (u + iKernel)*iKernelSize + (v+iKernel);
          T tPixelValue = pIn[iPixelIndex];
          iPixelValueX += tPixelValue*pKernelX[iKernelIndex];
          iPixelValueY += tPixelValue*pKernelY[iKernelIndex];

        }
        

      }
      
    }

    fPixelValue =
            sqrtf(
                static_cast<float>(iPixelValueX * iPixelValueX) +
                static_cast<float>(iPixelValueY * iPixelValueY)
            )/fKernelFactor;

    int iPixelIndex = Row*iWidth + Col;
    pOut[iPixelIndex] = static_cast<U>(fPixelValue);
    
  }
}

namespace ImgProc{

  namespace Edge
  {
    template <typename T, typename U>
    SobelFilter<T, U>::SobelFilter(
          std::size_t iWidth /*= 1*/,
          std::size_t iHeight /*= 32*/,
          int iThreads /*= 16*/,
          std::size_t iKernelSize/* = 3*/)
      
      : m_iWidth(iWidth)
      , m_iHeight(iHeight)
      , m_iThreads(iThreads)
      , m_iKernelSize(iKernelSize)
      , m_fNormalizingFactor(0.0f)
      , m_aKernelX(iKernelSize,iKernelSize, 1)
      , m_aKernelY(iKernelSize,iKernelSize, 1)
    {

      if( m_iKernelSize%2 == 0)
      {
        std::cout << "[WARNING] : A KERNEL SIZE SHOULD BE A ODD NUMBER SUP THAN 1, eg 3, 5, 7, 9, ..., 2*n+1" << std::endl;
      }

      createSobelKernel();
      m_iGridDimX = (m_iWidth  + m_iThreads - 1)/m_iThreads;
      m_iGridDimY = (m_iHeight + m_iThreads - 1)/m_iThreads;
    }


    template <typename T, typename U>
    void SobelFilter<T, U>::createSobelKernel()
    {

      auto binomial = [](int n, int k){
            if (k > n - k)
            {
              k = n - k;
            }

            int result = 1;

            for (int i = 1; i <= k; ++i)
            {
              result = result * (n - k + i) / i;
            }

            return result;
            
          };
      
          
      if (m_iKernelSize < 3 || m_iKernelSize % 2 == 0)
      {
        throw std::invalid_argument("m_iKernelSize must be odd and >= 3");
      }
      const int iCenter = (m_iKernelSize-1)/2;
      for(int i= 0; i < m_iKernelSize; ++i)
      {
        int iSmoothing = binomial( m_iKernelSize-1,i);
        for(int j=0; j < m_iKernelSize; ++j)
        {
          int iDerivative = j - iCenter;
          int iValue = iSmoothing * iDerivative;
          m_fNormalizingFactor +=static_cast<float>(std::abs(iValue));
          m_aKernelX(i,j) = iValue;
          m_aKernelY(j,i) = iValue;
        }
      }
      
      /*
      for(int i= 0; i < m_iKernelSize; ++i)
      {
      
        for(int j=0; j < m_iKernelSize; ++j)
        {
          std::cout << m_aKernelX(i,j) << " " ; 
        }
        std::cout  << "\n" ; 
      }
    
      //m_fNormalizingFactor = 1.0f;
      std::cout << m_fNormalizingFactor << " \n" ; 
      */
    }
    

    template <typename T, typename U>
    void SobelFilter<T, U>::KernelLauncher(const T *pIn_h, U *pOut_h)
    {
      std::cout << "Sobel Kernel Launcher " << std::endl;
      dim3 oGridDim(m_iGridDimX, m_iGridDimY);
      dim3 oBlockDim(m_iThreads, m_iThreads); 

      int iImageSize = m_iWidth * m_iHeight * sizeof(T);
      CudaBuffer<T> oCudaBufferIn(iImageSize);
      CudaBuffer<T> oCudaBufferOut(iImageSize);

      int iKernelSize = m_iKernelSize*m_iKernelSize*sizeof(int);
      CudaBuffer<int> oCudaBufferKernelX(iKernelSize);
      CudaBuffer<int> oCudaBufferKernelY(iKernelSize);


    
      oCudaBufferIn.copyFromHost(pIn_h);

      int* pSobelKernelX = m_aKernelX.data();
      oCudaBufferKernelX.copyFromHost(pSobelKernelX);

      int* pSobelKernelY = m_aKernelY.data();
      oCudaBufferKernelY.copyFromHost(pSobelKernelY);

      cudaEvent_t oStart, oStop;
      float fTimeInMS;

      CUDA_CALL(cudaEventCreate(&oStart));
      CUDA_CALL(cudaEventCreate(&oStop));
      CUDA_CALL(cudaEventRecord(oStart));

      SobelFilterKernel<T, U, int><<<oGridDim,oBlockDim>>>(oCudaBufferIn.Data(), 
                                                            oCudaBufferOut.Data(), 
                                                            oCudaBufferKernelX.Data(), 
                                                            oCudaBufferKernelY.Data(), 
                                                            m_iHeight, 
                                                            m_iWidth, 
                                                            m_iKernelSize,
                                                            m_fNormalizingFactor);

      CUDA_CALL(cudaEventRecord(oStop));
      CUDA_CALL(cudaEventSynchronize(oStop));
    
      CUDA_CALL(cudaEventElapsedTime(&fTimeInMS, oStart, oStop));
      std::cout << "CUDA kernel: " << fTimeInMS << "ms\n";

      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());

    
      oCudaBufferOut.copyFromDevice(pOut_h);
    }    

    template class SobelFilter<int, int>;
    template class SobelFilter<uchar, uchar>;
  } 
}
