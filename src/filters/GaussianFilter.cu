#include <iostream>
#include <stdio.h>
#include <cmath>
#include "ImgProc/filters/GaussianFilter.hpp"
#include "cuda/CudaBuffer.hpp"

using uchar = unsigned char;

template <class T, class U , class V>
__global__ void GaussianFilterKernel(const T* pIn, U* pOut, V* pKernel,  int iHeight, int iWidth, int iChannels =3, const std::size_t iKernelSize = 3 )
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(col < iWidth && row < iHeight )
    {
        V fPixelValue = 0.0f;
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

                    vBGR[0] += static_cast<float>(pIn[ iImageIndex     ]) * static_cast<float>(pKernel[iKernelIndex]) ;
                    vBGR[1] += static_cast<float>(pIn[ iImageIndex + 1 ]) * static_cast<float>(pKernel[iKernelIndex]) ;
                    vBGR[2] += static_cast<float>(pIn[ iImageIndex + 2 ]) * static_cast<float>(pKernel[iKernelIndex]) ;

                }
                
            }
            
            
        }

        int iIndex = (row*iWidth + col)*iChannels;
        pOut[ iIndex + 0 ] = static_cast<U>(vBGR[0]);
        pOut[ iIndex + 1 ] = static_cast<U>(vBGR[1]);
        pOut[ iIndex + 2 ] = static_cast<U>(vBGR[2]);

    }
}


namespace ImgProc
{
    namespace Filter{

        template <typename T, typename U>
        GaussianFilter<T,U>::GaussianFilter(std::size_t iWidth /*=1*/, std::size_t iHeight/*= 32*/,std::size_t iChannels /*=3*/,int iThreads /*=16*/, std::size_t iKernelSize/*=3*/, float fSigma /*=1.0f*/)
        : m_iWidth(iWidth)
        , m_iHeight(iHeight)
        , m_iChannels(iChannels)
        , m_iThreads(iThreads)
        , m_iKernelSize(iKernelSize)
        , m_fSigma(fSigma)
        , m_aKernel(iKernelSize, iKernelSize, 1)
        {
          std::cout << "Gaussian Filter" << std::endl; 
          if( m_iKernelSize%2 == 0)
          {
            std::cout << "[WARNING] : A KERNEL SIZE SHOULD BE A ODD NUMBER SUP THAN 1, eg 3, 5, 7, 9, ..., 2*n+1" << std::endl;
          }

          if( m_iKernelSize < 3 || m_iKernelSize % 2 == 0 )
          {
            throw std::invalid_argument("m_iKernelSize must be odd and >= 3");
          }

          createGaussianKernel();
          m_iGridDimX = (m_iWidth  + m_iThreads - 1) / m_iThreads;
          m_iGridDimY = (m_iHeight + m_iThreads - 1) / m_iThreads;
          std::cout << "( " << m_iGridDimX << " , " << m_iGridDimY << " , " << m_iThreads << " , " << m_iKernelSize << " )"<< std::endl;
        }
    
        template <typename T, typename U>
        void GaussianFilter<T,U>::createGaussianKernel()
        {   
    
            int iHalf = m_iKernelSize/2;
            //float fCenter = 3.0f*m_fSigma;
            float sum = 0.0f;
            for (int y = -iHalf; y <= iHalf; ++y)
            {
                for (int x = -iHalf; x <= iHalf; ++x)
                {
    
                    float fValue = (std::exp(-0.5f*(x*x + y*y)/(m_fSigma*m_fSigma)))*0.0f + 1.0f;
                    
                    m_aKernel(y+iHalf, x+iHalf) = fValue;
                    
                    sum += fValue;
                }
            }
    
            const float invSum = 1.0f / sum;
            for (size_t y = 0; y < m_iKernelSize; ++y)
            {
                for( size_t x = 0; x < m_iKernelSize; ++x )
                {
                    m_aKernel(y, x) *= invSum;
                }
            }
            
        }
    
        template <typename T, typename U>
        void GaussianFilter<T,U>::KernelLauncher(const T* pIn_h,  U* pOut_h)
        {
            std::cout << "Gaussian Kernel Launcher" << std::endl;
            
            dim3 oGridDim(m_iGridDimX, m_iGridDimY);
            dim3 oBlockDim(m_iThreads, m_iThreads);
    
            std::size_t iSize = m_iHeight*m_iWidth*m_iChannels * sizeof(U);
    
            CudaBuffer<T> oCudaBufferIn(iSize);
            CudaBuffer<U> oCudaBufferOut(iSize);
            CudaBuffer<float> oCudaBufferKernel(m_iKernelSize*m_iKernelSize*m_iChannels*sizeof(float));
           
            oCudaBufferIn.copyFromHost(pIn_h);
    
            float* pFloat = m_aKernel.data();
            oCudaBufferKernel.copyFromHost(pFloat);
            
            
             
            cudaEvent_t oStart, oStop;
            float fTimeInS;
    
            CUDA_CALL(cudaEventCreate(&oStart));
            CUDA_CALL(cudaEventCreate(&oStop));
            CUDA_CALL(cudaEventRecord(oStart));
            
            GaussianFilterKernel<T, U, float><<<oGridDim,oBlockDim>>>(oCudaBufferIn.Data(), oCudaBufferOut.Data(), oCudaBufferKernel.Data(), m_iHeight, m_iWidth, m_iChannels, m_iKernelSize );
            
            CUDA_CALL(cudaEventRecord(oStop));
            CUDA_CALL(cudaEventSynchronize(oStop));
      
            CUDA_CALL(cudaEventElapsedTime(&fTimeInS, oStart, oStop));
            std::cout << "CUDA kernel: " << fTimeInS << "ms\n";
    
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            
            oCudaBufferOut.copyFromDevice(pOut_h);
            
            
        }
        
        template class GaussianFilter<uchar, uchar>;
        template class GaussianFilter<int, float>;
        template class GaussianFilter<int, int>;
        template class GaussianFilter<float, int>;
        template class GaussianFilter<float, float>;
    }

}
