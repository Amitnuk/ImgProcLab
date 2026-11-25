#include "ImgProc/cuda/basicOperations/colorConversion/iplCuda.hpp"
#include "ImgProc/cuda/basicOperations/colorConversion/iplData.hpp"
#include "common/ImgProcLabUtils.hpp"
#include "cuda/common/CudaUtils.hpp"
#include <cmath>
#include <stdexcept>


namespace ImgProcLab::detail
{

    constexpr int CHANNELS = 3;
    ColorConversion prepareGrayConversion(const uchar* pImage,
                                          const std::size_t iHeight,
                                          const std::size_t iWidth)
    {

        if(!pImage)
        {
            throw std::runtime_error("Invalid Image");
        }

        if(iHeight <= 0 || iWidth <= 0)
        {
            throw std::runtime_error("Invalid Image Size");
        }

        ColorConversion eData;

        std::size_t iImageInBytesIn = iWidth * iHeight * CHANNELS * sizeof(uchar);
        std::size_t iImageInBytesOut = iWidth * iHeight * sizeof(uchar);
        try
        {
            eData.m_eCudaExecutionContext.m_oDeviceInput.allocate(iImageInBytesIn);
            eData.m_eCudaExecutionContext.m_oDeviceOutputs.allocate(iImageInBytesOut);
            eData.m_eCudaExecutionContext.m_iHeight = iHeight;
            eData.m_eCudaExecutionContext.m_iWidth = iWidth;
            eData.m_eCudaExecutionContext.m_oDeviceInput.copyFromHost(pImage);
        }
        catch(const std::exception& e)
        {
            std::fprintf(stderr, "Cuda error %s\n: ", e.what());
            throw std::invalid_argument("Failed to to prepare Gaussian");
        }

        return eData;
    }

} // namespace ImgProcLab::detail
