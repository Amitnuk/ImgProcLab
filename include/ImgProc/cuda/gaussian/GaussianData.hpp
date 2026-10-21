#ifndef GAUSSIAN_DATA_HPP
#define GAUSSIAN_DATA_HPP
#include "cuda/common/CudaExecutionContext.hpp"
#include "math/Array3D.hpp"

namespace ImgProcLab::detail
{
    struct GaussianData
    {
        int m_iKernelSize = 0;
        float m_fSigma = 0.0f;
        Array<float> m_aCoeficients;

        CudaExecutionContext m_eCudaExecutionContext;
        
    };
} 

#endif
