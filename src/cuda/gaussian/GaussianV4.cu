#include "ImgProc/cuda/gaussian/GaussianCuda.hpp"
#include "ImgProc/cuda/gaussian/GaussianData.hpp"
#include "common/ImgProcLabUtils.hpp"
#include "cuda/common/CudaExecutionContext.hpp"
#include <cstddef>
#include <stdexcept>


namespace ImgProcLab
{
namespace detail
{
constexpr int CHANNELS = 3;
__global__ void GaussianFilterKernelV4(const uchar *pIn, uchar *pOut, const float *pKernel, int iHeight, int iWidth, const std::size_t iKernelSize = 3)
{

    int col = threadIdx.x + blockIdx.x * CONSTANTS::DEFAULT_THREADS;
    int row = threadIdx.y + blockIdx.y * CONSTANTS::DEFAULT_THREADS; //

    __shared__ float Mds[CONSTANTS::DEFAULT_THREADS * CONSTANTS::DEFAULT_THREADS * CHANNELS];


    // load Mds --> thread block
    if((row < iHeight) && (col < iWidth))
    {
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 0] = static_cast<float>(pIn[(row * iWidth + col) * CHANNELS + 0]);
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 1] = static_cast<float>(pIn[(row * iWidth + col) * CHANNELS + 1]);
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 2] = static_cast<float>(pIn[(row * iWidth + col) * CHANNELS + 2]);
    }
    else
    {
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 0] = 0.0f;
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 1] = 0.0f;
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 2] = 0.0f;
    }

    __syncthreads();


    if((row < iHeight) && (col < iWidth))
    {

        float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};
        int iRadius = iKernelSize / 2;

        for(int iRow = 0; iRow < iKernelSize; ++iRow)
        {
            for(int iCol = 0; iCol < iKernelSize; ++iCol)
            {
                if((static_cast<int>(threadIdx.x) + iCol - iRadius >= 0) &&
                   (static_cast<int>(threadIdx.x) + iCol - iRadius < CONSTANTS::DEFAULT_THREADS) &&
                   (static_cast<int>(threadIdx.y) + iRow - iRadius >= 0) &&
                   (static_cast<int>(threadIdx.y) + iRow - iRadius < CONSTANTS::DEFAULT_THREADS))
                { // inside a block
                    int iImageIndex = ((threadIdx.y - iRadius + iRow) * CONSTANTS::DEFAULT_THREADS + (threadIdx.x - iRadius + iCol)) * CHANNELS;
                    int iKernelIndex = iRow * iKernelSize + iCol;
                    const float &fWeights = pKernel[iKernelIndex];
                    vBGR[0] += Mds[iImageIndex + 0] * fWeights;
                    vBGR[1] += Mds[iImageIndex + 1] * fWeights;
                    vBGR[2] += Mds[iImageIndex + 2] * fWeights;
                }
                else if((row + iRow - iRadius >= 0) &&
                        (row + iRow - iRadius < iHeight) &&
                        (col + iCol - iRadius >= 0) &&
                        (col + iCol - iRadius < iWidth))
                { // outside a block but still inside an image --> halo cells
                    int iImageIndex = ((row - iRadius + iRow) * iWidth + (col - iRadius + iCol)) * CHANNELS;
                    int iKernelIndex = iRow * iKernelSize + iCol;
                    const float &fWeights = pKernel[iKernelIndex];
                    vBGR[0] += pIn[iImageIndex + 0] * fWeights;
                    vBGR[1] += pIn[iImageIndex + 1] * fWeights;
                    vBGR[2] += pIn[iImageIndex + 2] * fWeights;
                }
            }
        }

        int iImageIndex = (row * iWidth + col) * CHANNELS;
        pOut[iImageIndex + 0] = static_cast<uchar>(__float2int_rn(vBGR[0]));
        pOut[iImageIndex + 1] = static_cast<uchar>(__float2int_rn(vBGR[1]));
        pOut[iImageIndex + 2] = static_cast<uchar>(__float2int_rn(vBGR[2]));
    }
}

__global__ void GaussianFilterKernelHorizontalV4(const uchar *pIn, float *pOut, const float *pKernel, int iHeight, int iWidth, const std::size_t iKernelSize = 3)
{
    int col = threadIdx.x + blockIdx.x * CONSTANTS::DEFAULT_THREADS;
    int row = threadIdx.y + blockIdx.y * CONSTANTS::DEFAULT_THREADS;

    __shared__ float Mds[CONSTANTS::DEFAULT_THREADS * CONSTANTS::DEFAULT_THREADS * CHANNELS];


    if(row < iHeight && col < iWidth)
    {
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 0] = static_cast<float>(pIn[(row * iWidth + col) * CHANNELS + 0]);
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 1] = static_cast<float>(pIn[(row * iWidth + col) * CHANNELS + 1]);
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 2] = static_cast<float>(pIn[(row * iWidth + col) * CHANNELS + 2]);
    }
    else
    {
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 0] = 0.0f;
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 1] = 0.0f;
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 2] = 0.0f;
    }


    __syncthreads();


    if(col < iWidth && row < iHeight)
    {
        float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};
        int iRadius = iKernelSize / 2;


        for(int iCol = 0; iCol < iKernelSize; ++iCol)
        {
            if(static_cast<int>(threadIdx.x) + iCol - iRadius >= 0 &&
               static_cast<int>(threadIdx.x) + iCol - iRadius < CONSTANTS::DEFAULT_THREADS)
            {
                int iImageIndex = (threadIdx.y * CONSTANTS::DEFAULT_THREADS + (threadIdx.x + iCol - iRadius)) * CHANNELS;
                const float &fWeights = pKernel[iCol];
                vBGR[0] += Mds[iImageIndex + 0] * fWeights;
                vBGR[1] += Mds[iImageIndex + 1] * fWeights;
                vBGR[2] += Mds[iImageIndex + 2] * fWeights;
            }
            else if(col + iCol - iRadius >= 0 &&
                    col + iCol - iRadius < iWidth)
            {

                int iImageIndex = (row * iWidth + (col + iCol - iRadius)) * CHANNELS;
                const float &fWeights = pKernel[iCol];
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

__global__ void GaussianFilterKernelVerticalV4(const float *pIn, uchar *pOut, const float *pKernel, int iHeight, int iWidth, const std::size_t iKernelSize = 3)
{
    int col = threadIdx.x + blockIdx.x * CONSTANTS::DEFAULT_THREADS;
    int row = threadIdx.y + blockIdx.y * CONSTANTS::DEFAULT_THREADS;

    __shared__ float Mds[CONSTANTS::DEFAULT_THREADS * CONSTANTS::DEFAULT_THREADS * CHANNELS];


    if(row < iHeight && col < iWidth)
    {
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 0] = pIn[(row * iWidth + col) * CHANNELS + 0];
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 1] = pIn[(row * iWidth + col) * CHANNELS + 1];
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 2] = pIn[(row * iWidth + col) * CHANNELS + 2];
    }
    else
    {
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 0] = 0.0f;
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 1] = 0.0f;
        Mds[(threadIdx.y * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS + 2] = 0.0f;
    }


    __syncthreads();


    if(col < iWidth && row < iHeight)
    {
        float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};
        int iRadius = iKernelSize / 2;


        for(int iRow = 0; iRow < iKernelSize; ++iRow)
        {
            if(static_cast<int>(threadIdx.y) + iRow - iRadius >= 0 &&
               static_cast<int>(threadIdx.y) + iRow - iRadius < CONSTANTS::DEFAULT_THREADS)
            {
                int iImageIndex = ((threadIdx.y + iRow - iRadius) * CONSTANTS::DEFAULT_THREADS + threadIdx.x) * CHANNELS;
                const float &fWeights = pKernel[iRow];
                vBGR[0] += Mds[iImageIndex + 0] * fWeights;
                vBGR[1] += Mds[iImageIndex + 1] * fWeights;
                vBGR[2] += Mds[iImageIndex + 2] * fWeights;
            }
            else if(row + iRow - iRadius >= 0 &&
                    row + iRow - iRadius < iHeight)
            {

                int iImageIndex = ((row + iRow - iRadius) * iWidth + col) * CHANNELS;
                const float &fWeights = pKernel[iRow];
                vBGR[0] += static_cast<float>(pIn[iImageIndex + 0]) * fWeights;
                vBGR[1] += static_cast<float>(pIn[iImageIndex + 1]) * fWeights;
                vBGR[2] += static_cast<float>(pIn[iImageIndex + 2]) * fWeights;
            }
        }
        int iImageIndex = (row * iWidth + col) * CHANNELS;
        pOut[iImageIndex + 0] = static_cast<uchar>(__float2int_rn(vBGR[0]));
        pOut[iImageIndex + 1] = static_cast<uchar>(__float2int_rn(vBGR[1]));
        pOut[iImageIndex + 2] = static_cast<uchar>(__float2int_rn(vBGR[2]));
    }
}

void gaussianV4(GaussianData &eData, const int iThreads /*=CONSTANTS::DEFAULT_THREADS*/, const Separable eCase /*= Separable::FALSE*/)
{

    CudaExecutionContext &eCudaContext = eData.m_eCudaExecutionContext;

    const int &iHeight = eCudaContext.m_iHeight;
    const int &iWidth = eCudaContext.m_iWidth;

    int iGridDimX = (iWidth + iThreads - 1) / iThreads;
    int iGridDimY = (iHeight + iThreads - 1) / iThreads;

    dim3 oGridDim(iGridDimX, iGridDimY);
    dim3 oBlockDim(iThreads, iThreads);

    if(eCase == Separable::FALSE)
    {
        GaussianFilterKernelV4<<<oGridDim, oBlockDim>>>(eCudaContext.m_oDeviceInput,
                                                        eCudaContext.m_oDeviceOutputs,
                                                        eCudaContext.m_oKernel.Data(),
                                                        iHeight,
                                                        iWidth,
                                                        eData.m_iKernelSize);
    }
    else if(eCase == Separable::TRUE)
    {
        GaussianFilterKernelHorizontalV4<<<oGridDim, oBlockDim>>>(eCudaContext.m_oDeviceInput,
                                                                  eCudaContext.m_oDeviceIntermediate,
                                                                  eCudaContext.m_oKernel.Data(),
                                                                  iHeight,
                                                                  iWidth,
                                                                  eData.m_iKernelSize);

        GaussianFilterKernelVerticalV4<<<oGridDim, oBlockDim>>>(eCudaContext.m_oDeviceIntermediate,
                                                                eCudaContext.m_oDeviceOutputs,
                                                                eCudaContext.m_oKernel.Data(),
                                                                iHeight,
                                                                iWidth,
                                                                eData.m_iKernelSize);
    }
    else
    {
        throw std::invalid_argument("Unknown version of the Gaussian");
    }
}
} // namespace detail
} // namespace ImgProcLab
