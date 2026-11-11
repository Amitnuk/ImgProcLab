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
        extern __constant__ float GFILTER[CONSTANTS::MAX_KERNEL_SIZE * CONSTANTS::MAX_KERNEL_SIZE];
        constexpr int CHANNELS = 3;
        __global__ void GaussianFilterKernelV5(const uchar *pIn, uchar *pOut, int iHeight, int iWidth, const std::size_t iKernelSize = 3)
        {

            int col = threadIdx.x + blockIdx.x * CONSTANTS::BLOCK_DIM_X;
            int row = threadIdx.y + blockIdx.y * CONSTANTS::BLOCK_DIM_Y; //

            __shared__ float Mds[CONSTANTS::BLOCK_DIM_X * CONSTANTS::BLOCK_DIM_Y * CHANNELS];


            // load Mds --> thread block
            if((row < iHeight) && (col < iWidth))
            {
                Mds[(threadIdx.y * CONSTANTS::BLOCK_DIM_X + threadIdx.x) * CHANNELS + 0] = static_cast<float>(pIn[(row * iWidth + col) * CHANNELS + 0]);
                Mds[(threadIdx.y * CONSTANTS::BLOCK_DIM_X + threadIdx.x) * CHANNELS + 1] = static_cast<float>(pIn[(row * iWidth + col) * CHANNELS + 1]);
                Mds[(threadIdx.y * CONSTANTS::BLOCK_DIM_X + threadIdx.x) * CHANNELS + 2] = static_cast<float>(pIn[(row * iWidth + col) * CHANNELS + 2]);
            }
            else
            {
                Mds[(threadIdx.y * CONSTANTS::BLOCK_DIM_X + threadIdx.x) * CHANNELS + 0] = 0.0f;
                Mds[(threadIdx.y * CONSTANTS::BLOCK_DIM_X + threadIdx.x) * CHANNELS + 1] = 0.0f;
                Mds[(threadIdx.y * CONSTANTS::BLOCK_DIM_X + threadIdx.x) * CHANNELS + 2] = 0.0f;
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
                           (static_cast<int>(threadIdx.x) + iCol - iRadius < CONSTANTS::BLOCK_DIM_X) &&
                           (static_cast<int>(threadIdx.y) + iRow - iRadius >= 0) &&
                           (static_cast<int>(threadIdx.y) + iRow - iRadius < CONSTANTS::BLOCK_DIM_Y))
                        { // inside a block
                            int iImageIndex = ((threadIdx.y - iRadius + iRow) * CONSTANTS::BLOCK_DIM_X + (threadIdx.x - iRadius + iCol)) * CHANNELS;
                            int iKernelIndex = iRow * iKernelSize + iCol;
                            const float &fWeights = GFILTER[iKernelIndex];
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
                            const float &fWeights = GFILTER[iKernelIndex];
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

        __global__ void GaussianFilterKernelHorizontalV5(const uchar *pIn, float *pOut, int iHeight, int iWidth, const std::size_t iKernelSize = 3)
        {

            extern __shared__ float Mds[];
            int iRadius = iKernelSize / 2;
            const int BLOCK = CONSTANTS::BLOCK_DIM_X + 2 * iRadius;
            int col = threadIdx.x + blockIdx.x * CONSTANTS::BLOCK_DIM_X;
            int row = threadIdx.y + blockIdx.y * CONSTANTS::BLOCK_DIM_Y;
            {

                int iSharedIndex = (threadIdx.y * BLOCK + threadIdx.x) * CHANNELS;

                if(row < iHeight && col - iRadius >= 0 && col - iRadius < iWidth)
                {
                    int iInputIndex = (row * iWidth - iRadius + col) * CHANNELS;
                    Mds[iSharedIndex + 0] = static_cast<float>(pIn[iInputIndex + 0]);
                    Mds[iSharedIndex + 1] = static_cast<float>(pIn[iInputIndex + 1]);
                    Mds[iSharedIndex + 2] = static_cast<float>(pIn[iInputIndex + 2]);
                }
                else
                {
                    Mds[iSharedIndex + 0] = 0.0f;
                    Mds[iSharedIndex + 1] = 0.0f;
                    Mds[iSharedIndex + 2] = 0.0f;
                }
            }

            if(threadIdx.x < 2 * iRadius)
            {


                int iInputIndex = (row * iWidth - iRadius + col + CONSTANTS::BLOCK_DIM_X) * CHANNELS;
                int iSharedIndex = (threadIdx.y * BLOCK + threadIdx.x + CONSTANTS::BLOCK_DIM_X) * CHANNELS;

                if(row < iHeight && col - iRadius + CONSTANTS::BLOCK_DIM_X >= 0 && col - iRadius + CONSTANTS::BLOCK_DIM_X < iWidth)
                {

                    Mds[iSharedIndex + 0] = static_cast<float>(pIn[iInputIndex + 0]);
                    Mds[iSharedIndex + 1] = static_cast<float>(pIn[iInputIndex + 1]);
                    Mds[iSharedIndex + 2] = static_cast<float>(pIn[iInputIndex + 2]);
                }
                else
                {
                    Mds[iSharedIndex + 0] = 0.0f;
                    Mds[iSharedIndex + 1] = 0.0f;
                    Mds[iSharedIndex + 2] = 0.0f;
                }
            }
            __syncthreads();

            if(col < iWidth && row < iHeight)
            {
                float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};


                for(int iCol = 0; iCol < iKernelSize; ++iCol)
                {

                    int iImageIndex = (threadIdx.y * BLOCK + (threadIdx.x + iCol)) * CHANNELS;
                    const float &fWeights = GFILTER[iCol];
                    vBGR[0] += Mds[iImageIndex + 0] * fWeights;
                    vBGR[1] += Mds[iImageIndex + 1] * fWeights;
                    vBGR[2] += Mds[iImageIndex + 2] * fWeights;
                }
                int iImageIndex = (row * iWidth + col) * CHANNELS;
                pOut[iImageIndex + 0] = vBGR[0];
                pOut[iImageIndex + 1] = vBGR[1];
                pOut[iImageIndex + 2] = vBGR[2];
            }
        }

        __global__ void GaussianFilterKernelVerticalV5(const float *pIn, uchar *pOut, int iHeight, int iWidth, const std::size_t iKernelSize = 3)
        {
            extern __shared__ float Mds[];

            int iRadius = iKernelSize / 2;
            const int BLOCK = CONSTANTS::BLOCK_DIM_X;

            int col = threadIdx.x + blockIdx.x * CONSTANTS::BLOCK_DIM_X;
            int row = threadIdx.y + blockIdx.y * CONSTANTS::BLOCK_DIM_Y;

            // First load: shared-memory rows 0–7
            {
                int iSharedIndex =
                    (threadIdx.y * BLOCK + threadIdx.x) * CHANNELS;

                int iInputRow = row - iRadius;

                if(col < iWidth && iInputRow >= 0 && iInputRow < iHeight)
                {
                    int iInputIndex = (iInputRow * iWidth + col) * CHANNELS;

                    Mds[iSharedIndex + 0] = pIn[iInputIndex + 0];
                    Mds[iSharedIndex + 1] = pIn[iInputIndex + 1];
                    Mds[iSharedIndex + 2] = pIn[iInputIndex + 2];
                }
                else
                {
                    Mds[iSharedIndex + 0] = 0.0f;
                    Mds[iSharedIndex + 1] = 0.0f;
                    Mds[iSharedIndex + 2] = 0.0f;
                }
            }

            // Second load: shared-memory rows 8–15
            if(threadIdx.y < 2 * iRadius)
            {
                int iSharedIndex =
                    ((threadIdx.y + CONSTANTS::BLOCK_DIM_Y) * BLOCK + threadIdx.x) * CHANNELS;

                int iInputRow = row - iRadius + CONSTANTS::BLOCK_DIM_Y;

                if(col < iWidth && iInputRow >= 0 && iInputRow < iHeight)
                {
                    int iInputIndex = (iInputRow * iWidth + col) * CHANNELS;

                    Mds[iSharedIndex + 0] = pIn[iInputIndex + 0];
                    Mds[iSharedIndex + 1] = pIn[iInputIndex + 1];
                    Mds[iSharedIndex + 2] = pIn[iInputIndex + 2];
                }
                else
                {
                    Mds[iSharedIndex + 0] = 0.0f;
                    Mds[iSharedIndex + 1] = 0.0f;
                    Mds[iSharedIndex + 2] = 0.0f;
                }
            }

            // Third load: shared-memory rows 16–23
            if(threadIdx.y + CONSTANTS::BLOCK_DIM_Y < 2 * iRadius)
            {
                int iSharedIndex =
                    ((threadIdx.y + CONSTANTS::BLOCK_DIM_Y * 2) * BLOCK + threadIdx.x) * CHANNELS;

                int iInputRow = row - iRadius + CONSTANTS::BLOCK_DIM_Y * 2;

                if(col < iWidth && iInputRow >= 0 && iInputRow < iHeight)
                {
                    int iInputIndex = (iInputRow * iWidth + col) * CHANNELS;

                    Mds[iSharedIndex + 0] = pIn[iInputIndex + 0];
                    Mds[iSharedIndex + 1] = pIn[iInputIndex + 1];
                    Mds[iSharedIndex + 2] = pIn[iInputIndex + 2];
                }
                else
                {
                    Mds[iSharedIndex + 0] = 0.0f;
                    Mds[iSharedIndex + 1] = 0.0f;
                    Mds[iSharedIndex + 2] = 0.0f;
                }
            }

            // Fourth load: shared-memory rows 24–31
            if(threadIdx.y + CONSTANTS::BLOCK_DIM_Y * 2 < 2 * iRadius)
            {
                int iSharedIndex =
                    ((threadIdx.y + CONSTANTS::BLOCK_DIM_Y * 3) * BLOCK + threadIdx.x) * CHANNELS;

                int iInputRow = row - iRadius + CONSTANTS::BLOCK_DIM_Y * 3;

                if(col < iWidth && iInputRow >= 0 && iInputRow < iHeight)
                {
                    int iInputIndex = (iInputRow * iWidth + col) * CHANNELS;

                    Mds[iSharedIndex + 0] = pIn[iInputIndex + 0];
                    Mds[iSharedIndex + 1] = pIn[iInputIndex + 1];
                    Mds[iSharedIndex + 2] = pIn[iInputIndex + 2];
                }
                else
                {
                    Mds[iSharedIndex + 0] = 0.0f;
                    Mds[iSharedIndex + 1] = 0.0f;
                    Mds[iSharedIndex + 2] = 0.0f;
                }
            }

            // Fifth load: shared-memory rows 32–37
            if(threadIdx.y + CONSTANTS::BLOCK_DIM_Y * 3 < 2 * iRadius)
            {
                int iSharedIndex =
                    ((threadIdx.y + CONSTANTS::BLOCK_DIM_Y * 4) * BLOCK + threadIdx.x) * CHANNELS;

                int iInputRow = row - iRadius + CONSTANTS::BLOCK_DIM_Y * 4;

                if(col < iWidth && iInputRow >= 0 && iInputRow < iHeight)
                {
                    int iInputIndex = (iInputRow * iWidth + col) * CHANNELS;

                    Mds[iSharedIndex + 0] = pIn[iInputIndex + 0];
                    Mds[iSharedIndex + 1] = pIn[iInputIndex + 1];
                    Mds[iSharedIndex + 2] = pIn[iInputIndex + 2];
                }
                else
                {
                    Mds[iSharedIndex + 0] = 0.0f;
                    Mds[iSharedIndex + 1] = 0.0f;
                    Mds[iSharedIndex + 2] = 0.0f;
                }
            }

            __syncthreads();

            if(col < iWidth && row < iHeight)
            {
                float vBGR[CHANNELS] = {0.0f, 0.0f, 0.0f};

                for(int iRow = 0; iRow < iKernelSize; ++iRow)
                {
                    int iImageIndex =
                        ((threadIdx.y + iRow) * BLOCK + threadIdx.x) * CHANNELS;

                    const float &fWeights = GFILTER[iRow];

                    vBGR[0] += Mds[iImageIndex + 0] * fWeights;
                    vBGR[1] += Mds[iImageIndex + 1] * fWeights;
                    vBGR[2] += Mds[iImageIndex + 2] * fWeights;
                }

                int iImageIndex = (row * iWidth + col) * CHANNELS;

                pOut[iImageIndex + 0] = static_cast<uchar>(__float2int_rn(vBGR[0]));
                pOut[iImageIndex + 1] = static_cast<uchar>(__float2int_rn(vBGR[1]));
                pOut[iImageIndex + 2] = static_cast<uchar>(__float2int_rn(vBGR[2]));
            }
        }

        void gaussianV5(GaussianData &eData, const Separable eCase /*= Separable::FALSE*/)
        {

            CudaExecutionContext &eCudaContext = eData.m_eCudaExecutionContext;

            const int &iHeight = eCudaContext.m_iHeight;
            const int &iWidth = eCudaContext.m_iWidth;

            int iGridDimX = (iWidth + CONSTANTS::BLOCK_DIM_X - 1) / CONSTANTS::BLOCK_DIM_X;
            int iGridDimY = (iHeight + CONSTANTS::BLOCK_DIM_Y - 1) / CONSTANTS::BLOCK_DIM_Y;

            dim3 oGridDim(iGridDimX, iGridDimY);
            dim3 oBlockDim(CONSTANTS::BLOCK_DIM_X, CONSTANTS::BLOCK_DIM_Y);

            int iRadius = eData.m_iKernelSize / 2;
            std::size_t iHozintalBytes = (CONSTANTS::BLOCK_DIM_X + 2 * iRadius) * CONSTANTS::BLOCK_DIM_Y * CHANNELS * sizeof(float);
            std::size_t iVerticalBytes = CONSTANTS::BLOCK_DIM_X * (CONSTANTS::BLOCK_DIM_Y + 2 * iRadius) * CHANNELS * sizeof(float);

            GaussianFilterKernelHorizontalV5<<<oGridDim, oBlockDim, iHozintalBytes>>>(eCudaContext.m_oDeviceInput,
                                                                                      eCudaContext.m_oDeviceIntermediate,
                                                                                      iHeight,
                                                                                      iWidth,
                                                                                      eData.m_iKernelSize);

            GaussianFilterKernelVerticalV5<<<oGridDim, oBlockDim, iVerticalBytes>>>(eCudaContext.m_oDeviceIntermediate,
                                                                                    eCudaContext.m_oDeviceOutputs,
                                                                                    iHeight,
                                                                                    iWidth,
                                                                                    eData.m_iKernelSize);
        }
    } // namespace detail
} // namespace ImgProcLab
