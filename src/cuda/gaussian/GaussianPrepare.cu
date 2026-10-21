#include <cuda_runtime.h>
#include <stdexcept>
#include "ImgProc/cuda/gaussian/GaussianData.hpp"
#include "ImgProc/cuda/gaussian/GaussianCuda.hpp"
#include <cmath>


namespace ImgProcLab::detail
{
    
    constexpr int CHANNELS = 3;

    void create2DGaussianKernel(float *pCoeficients, int iKernelSize, float fSigma)
    {   
        
      const float fInvTwoSigmaSquared = -0.5f / std::pow(fSigma,2);
        int iHalf = iKernelSize/2;
        float sum = 0.0f;
        float fValue = 0.0f;
        for (int y = -iHalf; y <= iHalf; ++y)
        {
            for (int x = -iHalf; x <= iHalf; ++x)
            {

                fValue = (std::exp((std::pow(x, 2) + std::pow(y,2))*fInvTwoSigmaSquared));
                pCoeficients[(y+iHalf)*iKernelSize + x+iHalf] = fValue;
            
                sum += fValue;
            }
        }

        const float fInvSum = 1.0f / sum;
        for (int y = 0; y < iKernelSize; ++y)
        {
            for( int x = 0; x < iKernelSize; ++x )
            {
                pCoeficients[y*iKernelSize + x] *= fInvSum;
            }
        }
        
    
    }
  
    void create1DGaussianKernel(float *pCoeficients, int iKernelSize, float fSigma)
    {
        const float fInvTwoSigmaSquared = -0.5f/std::pow(fSigma,2);
        int iHalf = iKernelSize/2;
        float fSum = 0.0f;
        float fValue = 0.0f;
        for(int i = -iHalf; i <= iHalf; ++i)
        {
            fValue = std::exp(std::pow(i, 2) * fInvTwoSigmaSquared);
            pCoeficients[(i+iHalf)] = fValue;
            fSum += fValue;
        }

        const float fInvSum = 1.0f/fSum; 
        for(int i = 0; i < iKernelSize; ++i)
        {
            pCoeficients[i] *= fInvSum;
        }

    }


  
  GaussianData prepareGaussian(const uchar* pImage,
                               const float fSigma,
                               const std::size_t iKernelSize ,
                               const std::size_t iHeight,
                               const std::size_t iWidth,
                               const Separable eCase /*= Separable::FALSE*/)
    {


        if(!pImage)
        {
            throw std::invalid_argument("Invalid Image");
        }
        
        if(iHeight <= 0 || iWidth <=0 )
        {
            throw std::invalid_argument("Invalid Image Size");
        }

        if( !std::isfinite(fSigma) || fSigma <= 0.0f || iKernelSize%2==0 || iKernelSize <= 0 )
        {
            throw std::invalid_argument("Invalid Gaussian parameters");
        }
        
        GaussianData eData; 
        eData.m_iKernelSize = iKernelSize;
        eData.m_fSigma = fSigma;

        eData.m_aCoeficients = Array<float>(iKernelSize, iKernelSize);
        if(eCase == Separable::FALSE)
        {
            create2DGaussianKernel(eData.m_aCoeficients.data(), iKernelSize, fSigma);
        }
        else
        {
          create1DGaussianKernel(eData.m_aCoeficients.data(), iKernelSize, fSigma);
        }

        
        
        std::size_t iImageInBytes = iWidth * iHeight * CHANNELS * sizeof(uchar);
        std::size_t iKernelSizeInBytes = iKernelSize * iKernelSize * sizeof(float);

        try
        {
            eData.m_eCudaExecutionContext.m_oDeviceInput.allocate(iImageInBytes);
            eData.m_eCudaExecutionContext.m_oDeviceOutputs.allocate(iImageInBytes);
            if(eCase == Separable::TRUE)
            {
              eData.m_eCudaExecutionContext.m_oDeviceIntermediate.allocate(iWidth * iHeight * CHANNELS * sizeof(float));
            }
            eData.m_eCudaExecutionContext.m_iHeight = iHeight;
            eData.m_eCudaExecutionContext.m_iWidth = iWidth;
            eData.m_eCudaExecutionContext.m_oDeviceInput.copyFromHost(pImage);
           
            eData.m_eCudaExecutionContext.m_oKernel.allocate(iKernelSizeInBytes);
            float* value = eData.m_aCoeficients.data();
            eData.m_eCudaExecutionContext.m_oKernel.copyFromHost(value);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            throw std::invalid_argument("Failed to to prepare Gaussian");
        }
        
        return eData;
    }

}

