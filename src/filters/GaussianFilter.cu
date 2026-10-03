#include <iostream>
#include <stdio.h>
#include <cmath>
#include "imgproc/filters/GaussianFilter.hpp"
#include "cuda/CudaBuffer.hpp"

using uchar = unsigned char;

__device__ int maxmax(int a, int b)
{
    return (a >b) ? a : b;
}

__device__ int minmin(int a, int b)
{
    return (a < b) ? a : b;
}


__global__ void GaussianFilterKernel(const uchar* pIn, uchar* pOut, uchar* pKernel,  int iHeight, int iWidth, int iChannels =3, int iKernelSize=3 )
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int depth = threadIdx.z ;

    if(col < iWidth && row < iHeight && depth < iChannels )
    {
        int iPixelValue = 0;
        int half = iKernelSize/2;
        
        for( int y = -half; y <= half; ++y )
        {
            for (int x = -half; x <= half; ++x)
            {
                int iCol = col + x;
                int iRow = row + y;
                
                if(( iCol < iWidth && iCol >= 0 ) && (iRow < iHeight && iRow >= 0))
                {
                    int iImageIndex  = (iRow*iWidth + iCol)*iChannels;
                    int iKernelIndex = iKernelSize*iKernelSize*depth + (y+half)*iKernelSize + (x + half);
                    iPixelValue += pIn[ iImageIndex + depth ]*pKernel[iKernelIndex] ;
                }
                
            }
            
            
        }

        int iIndex = (row*iWidth + col)*iChannels;
        iPixelValue = min(max(0, iPixelValue/256), 255);
        pOut[ iIndex + depth ] = static_cast<uchar>(iPixelValue);

    }
    



}


namespace ImgProc
{
    GaussianFilter::GaussianFilter(std::size_t iWidth /*=1*/, std::size_t iHeight/*= 32*/,std::size_t iChannels /*=3*/,int iThreads /*=16*/, std::size_t iKernelSize/*=3*/, float fSigma /*=1.0f*/)
    : m_iWidth(iWidth)
    , m_iHeight(iHeight)
    , m_iChannels(iChannels)
    , m_iThreads(iThreads)
    , m_iKernelSize(iKernelSize)
    , m_fSigma(fSigma)
    , m_aKernel(iKernelSize, iKernelSize, iChannels)
    {
      if( m_iKernelSize%2 == 0)
      {
        std::cout << "[WARNING] : A KERNEL SIZE SHOULD BE A ODD NUMBER SUP THAN 1, eg 3, 5, 7, 9, ..., 2*n+1" << std::endl;
      }
      createGaussianKernel();
      m_iGridDimX = (m_iWidth  + iThreads - 1) / iThreads;
      m_iGridDimY = (m_iHeight + iThreads - 1) / iThreads;
    }

    void GaussianFilter::createGaussianKernel()
    {   

        int iHalf = m_iKernelSize/2;
        float sum = 0.0f;
        for (int y = -iHalf; y <= iHalf; ++y)
        {
            for (int x = -iHalf; x <= iHalf; ++x)
            {
                //int index = (y+iHalf)*m_iSize + (x+iHalf);
                float fValue = std::exp(-(x*x + y*y)/(2.0f*m_fSigma*m_fSigma));
                
                //the same as having 3 1D kernels concatenated, forming a 3D
                for (int z = 0; z < m_iChannels; ++z)
                {
                    m_aKernel(y+iHalf, x+iHalf, z) = fValue;
                }
                
                sum += fValue;
            }
        }

        const float invSum = 1.0f / sum;
        for (size_t y = 0; y < m_iKernelSize; ++y)
        {
            for (size_t x = 0; x < m_iKernelSize; ++x)
            {
                for (int z = 0; z < m_iChannels; ++z)
                {

                    m_aKernel(y, x, z) *= invSum;
                }
            }
        }
        
    }

    void GaussianFilter::KernelLauncher(const unsigned char* pIn_h, unsigned char* pOut_h)
    {
        std::cout << "Kernel Launcher" << std::endl;
        
        dim3 oGridDim(m_iGridDimX, m_iGridDimY);
        dim3 oBlockDim(m_iThreads, m_iThreads, m_iChannels);

        std::size_t iSize = m_iHeight*m_iWidth*m_iChannels * sizeof(uchar);

        CudaBuffer<uchar> oCudaBufferIn(iSize);
        CudaBuffer<uchar> oCudaBufferOut(iSize);
        CudaBuffer<uchar> oCudaBufferKernel(m_iKernelSize*m_iKernelSize*m_iChannels*sizeof(uchar));
        
        oCudaBufferIn.copyFromHost(pIn_h);

        float* pFloat = m_aKernel.data();

        uchar* pUchar = new uchar[m_aKernel.size()];

        for (std::size_t i = 0; i < m_aKernel.size(); ++i)
        {
            pUchar[i] = static_cast<uchar>(pFloat[i] * 256.0f);
        }

        oCudaBufferKernel.copyFromHost(pUchar);
        
        
  
        
        
        std::cout << "Size = " << m_iKernelSize<< " " << m_iChannels << " "<< m_aKernel.height() <<" "<<m_aKernel.size() << std::endl;

        float fTotalX =0.0;
        float fTotalY =0.0;
        float fTotalZ =0.0;
        for(int i =0; i < m_aKernel.height(); i++)
        {
            for(int j =0; j < m_aKernel.height(); j++)
            {
                fTotalX += m_aKernel(i,j,0);
                fTotalY += m_aKernel(i,j,1);
                fTotalZ += m_aKernel(i,j,2);
            }
        }

        std::cout <<"sum :"<< fTotalX<< " " << fTotalY << " "<<fTotalZ << std::endl;
        cudaEvent_t oStart, oStop;
        float fTimeInS;

        CUDA_CALL(cudaEventCreate(&oStart));
        CUDA_CALL(cudaEventCreate(&oStop));
        CUDA_CALL(cudaEventRecord(oStart));

        GaussianFilterKernel<<<oGridDim,oBlockDim>>>(oCudaBufferIn.Data(), oCudaBufferOut.Data(), oCudaBufferKernel.Data(), m_iHeight, m_iWidth, m_iChannels, m_iKernelSize);
        
        CUDA_CALL(cudaEventRecord(oStop));
        CUDA_CALL(cudaEventSynchronize(oStop));
  
        CUDA_CALL(cudaEventElapsedTime(&fTimeInS, oStart, oStop));
        std::cout << "CUDA kernel: " << fTimeInS << " s\n";

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        
        oCudaBufferOut.copyFromDevice(pOut_h);
        delete[] pUchar;
    }
    

}