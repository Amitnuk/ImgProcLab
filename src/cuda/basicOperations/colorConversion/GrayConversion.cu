#include "ImgProc/cuda/basicOperations/colorConversion/iplCuda.hpp"
#include "ImgProc/cuda/basicOperations/colorConversion/iplData.hpp"
#include "common/ImgProcLabUtils.hpp"
#include "cuda/common/CudaExecutionContext.hpp"
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace ImgProcLab
{
    namespace detail
    {

        constexpr int CHANNELS = 3;
        __global__ void rgb2grayKernel(const uchar* pIn, uchar* pOut, int iHeight, int iWidth, const float r, const float g, const float b)
        {

            int row = threadIdx.y + blockIdx.y * blockDim.y;
            int col = threadIdx.x + blockIdx.x * blockDim.x;


            if(row < iHeight && col < iWidth)
            {
                int iColorIndex = (row * iWidth + col) * CHANNELS;

                float bChannel = pIn[iColorIndex];
                float gChannel = pIn[iColorIndex + 1];
                float rChannel = pIn[iColorIndex + 2];
                int iGrayIndex = (row * iWidth + col);
                pOut[iGrayIndex] = static_cast<uchar>(rChannel * r + g * gChannel + b * bChannel);
            }
        }

        void iplRGB2GRAY(ColorConversion& eData, const Separable eCase /*= Separable::FALSE*/)
        {

            CudaExecutionContext& eCudaContext = eData.m_eCudaExecutionContext;

            const int& iHeight = eCudaContext.m_iHeight;
            const int& iWidth = eCudaContext.m_iWidth;

            const int iGridDimX = (iWidth + CONSTANTS::BLOCK_DIM_X - 1) / CONSTANTS::BLOCK_DIM_X;
            const int iGridDimY = (iHeight + CONSTANTS::BLOCK_DIM_Y - 1) / CONSTANTS::BLOCK_DIM_Y;

            dim3 oGridDim(iGridDimX, iGridDimY);
            dim3 oBlockDim(CONSTANTS::BLOCK_DIM_X, CONSTANTS::BLOCK_DIM_Y);


            rgb2grayKernel<<<oGridDim, oBlockDim>>>(eCudaContext.m_oDeviceInput.Data(),
                                                    eCudaContext.m_oDeviceOutputs.Data(),
                                                    iHeight,
                                                    iWidth,
                                                    eData.r,
                                                    eData.g,
                                                    eData.b);
        }

    } // namespace detail
} // namespace ImgProcLab
