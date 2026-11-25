#pragma once
#include "ImgProc/cuda/basicOperations/colorConversion/iplData.hpp"
#include <opencv2/opencv.hpp>
#include <string_view>

namespace ImgProcLab
{

    namespace basicOps
    {

        class rgb2gray
        {
          public:
            // TODO: Optimize rgb2gray Constructor
            explicit rgb2gray(const cv::Mat oImage);


            rgb2gray(const rgb2gray&) = delete;
            rgb2gray& operator=(const rgb2gray&) = delete;

            rgb2gray(rgb2gray&& oRhs) noexcept
            : m_oInputImage(std::move(oRhs.m_oInputImage)),
              m_oOutputImage(std::move(oRhs.m_oOutputImage)),
              m_eColorConversionData(std::move(oRhs.m_eColorConversionData))
            {
            }

            rgb2gray& operator=(rgb2gray&& oRhs) noexcept
            {
                if(this == &oRhs)
                {
                    return *this;
                }

                m_oInputImage = std::move(oRhs.m_oInputImage);
                m_oOutputImage = std::move(oRhs.m_oOutputImage);
                m_eColorConversionData = std::move(oRhs.m_eColorConversionData);

                return *this;
            }

            ~rgb2gray() = default;
            // TODO:
            //  orgb2gray..getGaussianData() performs a move so orgb2gray does not have it anymore, should allow copy, Array should be copiable

            const detail::ColorConversion& getColorConversionData() const { return std::move(m_eColorConversionData); }; // const lvalue can be initialized with rvalue(1)
            detail::ColorConversion&& getColorConversionData() { return std::move(m_eColorConversionData); }             // rvalue can be initialized with rvalue(2)
                                                                                                                         // const detail::GaussianData& getGaussianData() {return m_eGaussianData;}; //const lvalue can be initialized with lvalue(3)
            const cv::Mat getImage() const { return m_oInputImage; }

          private:
            cv::Mat m_oInputImage;
            cv::Mat m_oOutputImage;
            detail::ColorConversion m_eColorConversionData;
        };
    } // namespace basicOps

} // namespace ImgProcLab
