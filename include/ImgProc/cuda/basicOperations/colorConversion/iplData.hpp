#ifndef GRAY_DATA_HPP
#define GRAY_DATA_HPP
#include "cuda/common/CudaExecutionContext.hpp"

namespace ImgProcLab::detail
{
    struct ColorConversion
    {
        float r = 0.299f;
        float g = 0.587f;
        float b = 0.114f;
        CudaExecutionContext m_eCudaExecutionContext;
    };
} // namespace ImgProcLab::detail

#endif
