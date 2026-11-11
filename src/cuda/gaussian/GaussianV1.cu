#include "ImgProc/cuda/gaussian/GaussianCuda.hpp"
#include "ImgProc/cuda/gaussian/GaussianData.hpp"
#include "common/ImgProcLabUtils.hpp"
#include "cuda/common/CudaExecutionContext.hpp"
#include <stdexcept>

namespace ImgProcLab
{
    namespace detail
    {
        constexpr int CHANNELS = 3;

        __global__ void GaussianFilterKernel(const uchar *pIn, uchar *pOut, const float *pKernel, int iHeight, int iWidth, const std::size_t iKernelSize = 3)
        {
            int col = threadIdx.x + blockIdx.x * blockDim.x;
            int row = threadIdx.y + blockIdx.y * blockDim.y;

            if(col < iWidth && row < iHeight)
            {
                int half = iKernelSize / 2;
                float vBGR[3] = {0.0f, 0.0f, 0.0f};
                for(int y = -half; y <= half; ++y)
                {
                    for(int x = -half; x <= half; ++x)
                    {
                        int iCol = col + x;
                        int iRow = row + y;

                        if((iCol < iWidth && iCol >= 0) && (iRow < iHeight && iRow >= 0))
                        {
                            int iImageIndex = (iRow * iWidth + iCol) * CHANNELS;
                            int iKernelIndex = (y + half) * iKernelSize + (x + half);

                            const float &fKernel = pKernel[iKernelIndex];
                            vBGR[0] += static_cast<float>(pIn[iImageIndex + 0]) * fKernel;
                            vBGR[1] += static_cast<float>(pIn[iImageIndex + 1]) * fKernel;
                            vBGR[2] += static_cast<float>(pIn[iImageIndex + 2]) * fKernel;
                        }
                    }
                }

                int iImageIndex = (row * iWidth + col) * CHANNELS;
                pOut[iImageIndex + 0] = static_cast<uchar>(__float2int_rn(vBGR[0]));
                pOut[iImageIndex + 1] = static_cast<uchar>(__float2int_rn(vBGR[1]));
                pOut[iImageIndex + 2] = static_cast<uchar>(__float2int_rn(vBGR[2]));
            }
        }


        void gaussianV1(GaussianData &eData, Separable eCase /*=Separable::FALSE*/)
        {

            CudaExecutionContext &eCudaContext = eData.m_eCudaExecutionContext;

            const int &iHeight = eCudaContext.m_iHeight;
            const int &iWidth = eCudaContext.m_iWidth;

            const int iGridDimX = (iWidth + CONSTANTS::BLOCK_DIM_X - 1) / CONSTANTS::BLOCK_DIM_X;
            const int iGridDimY = (iHeight + CONSTANTS::BLOCK_DIM_Y - 1) / CONSTANTS::BLOCK_DIM_Y;

            dim3 oGridDim(iGridDimX, iGridDimY);
            dim3 oBlockDim(CONSTANTS::BLOCK_DIM_X, CONSTANTS::BLOCK_DIM_Y);


            GaussianFilterKernel<<<oGridDim, oBlockDim>>>(eCudaContext.m_oDeviceInput.Data(),
                                                          eCudaContext.m_oDeviceOutputs.Data(),
                                                          eCudaContext.m_oKernel.Data(),
                                                          iHeight,
                                                          iWidth,
                                                          eData.m_iKernelSize);
        }

    } // namespace detail
} // namespace ImgProcLab
