#include "ImgProc/cuda/gaussian/GaussianCuda.hpp"
#include "ImgProc/cuda/gaussian/GaussianData.hpp"
#include "common/ImgProcLabUtils.hpp"
#include "cuda/common/CudaExecutionContext.hpp"
#include <stdexcept>

namespace ImgProcLab::detail
{

    constexpr int CHANNELS = 3;
    __global__ void GaussianFilterKernelHorizontalV2(const uchar *pIn,
                                                     float *pOut,
                                                     const float *pKernel,
                                                     int iHeight,
                                                     int iWidth,
                                                     const std::size_t iKernelSize = 3)
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;


        if(col < iWidth && row < iHeight)
        {
            int iHalf = iKernelSize / 2;
            float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};

            for(int x = -iHalf; x <= iHalf; ++x)
            {
                int iCol = col + x;
                if(iCol >= 0 && iCol < iWidth)
                {
                    int iKernelIndex = x + iHalf;
                    int iImageIndex = (row * iWidth + iCol) * CHANNELS;
                    const float &fWeights = pKernel[iKernelIndex];
                    vBGR[0] += static_cast<float>(pIn[iImageIndex + 0]) * fWeights;
                    vBGR[1] += static_cast<float>(pIn[iImageIndex + 1]) * fWeights;
                    vBGR[2] += static_cast<float>(pIn[iImageIndex + 2]) * fWeights;
                }
            }

            int iImageIndex = (row * iWidth + col) * CHANNELS;

            pOut[iImageIndex + 0] = vBGR[0];
            pOut[iImageIndex + 1] = vBGR[1];
            pOut[iImageIndex + 2] = vBGR[2];
        }
    }

    __global__ void GaussianFilterKernelVerticalV2(const float *pIn,
                                                   uchar *pOut,
                                                   const float *pKernel,
                                                   int iHeight,
                                                   int iWidth,
                                                   const std::size_t iKernelSize = 3)
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;


        if(col < iWidth && row < iHeight)
        {
            int iHalf = iKernelSize / 2;
            float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};

            for(int y = -iHalf; y <= iHalf; ++y)
            {
                int iRow = y + row;
                if(iRow >= 0 && iRow < iHeight)
                {
                    int iKernelIndex = y + iHalf;
                    int iInmageIndex = (iRow * iWidth + col) * CHANNELS;
                    const float &fWeights = pKernel[iKernelIndex];
                    vBGR[0] += pIn[iInmageIndex + 0] * fWeights;
                    vBGR[1] += pIn[iInmageIndex + 1] * fWeights;
                    vBGR[2] += pIn[iInmageIndex + 2] * fWeights;
                }
            }

            int iImageIndex = (row * iWidth + col) * CHANNELS;

            pOut[iImageIndex + 0] = static_cast<uchar>(__float2int_rn(vBGR[0]));
            pOut[iImageIndex + 1] = static_cast<uchar>(__float2int_rn(vBGR[1]));
            pOut[iImageIndex + 2] = static_cast<uchar>(__float2int_rn(vBGR[2]));
        }
    }

    void gaussianV2(GaussianData &eData, Separable eCase /*=Separable::FALSE*/)
    {

        CudaExecutionContext &eCudaContext = eData.m_eCudaExecutionContext;

        const int &iHeight = eCudaContext.m_iHeight;
        const int &iWidth = eCudaContext.m_iWidth;

        const int iGridDimX = (iWidth + CONSTANTS::BLOCK_DIM_X - 1) / CONSTANTS::BLOCK_DIM_X;
        const int iGridDimY = (iHeight + CONSTANTS::BLOCK_DIM_Y - 1) / CONSTANTS::BLOCK_DIM_Y;

        dim3 oGridDim(iGridDimX, iGridDimY);
        dim3 oBlockDim(CONSTANTS::BLOCK_DIM_X, CONSTANTS::BLOCK_DIM_Y);


        GaussianFilterKernelHorizontalV2<<<oGridDim, oBlockDim>>>(eCudaContext.m_oDeviceInput.Data(),
                                                                  eCudaContext.m_oDeviceIntermediate.Data(),
                                                                  eCudaContext.m_oKernel.Data(),
                                                                  iHeight,
                                                                  iWidth,
                                                                  eData.m_iKernelSize);


        GaussianFilterKernelVerticalV2<<<oGridDim, oBlockDim>>>(eCudaContext.m_oDeviceIntermediate.Data(),
                                                                eCudaContext.m_oDeviceOutputs.Data(),
                                                                eCudaContext.m_oKernel.Data(),
                                                                iHeight,
                                                                iWidth,
                                                                eData.m_iKernelSize);
    }

} // namespace ImgProcLab::detail
