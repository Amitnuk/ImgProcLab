#pragma once
#include "common/GaussianUtils.hpp"
#include "common/ImgProcLabUtils.hpp"
#include <cstddef>

using uchar = unsigned char;

namespace ImgProcLab
{
    namespace detail
    {
        class ColorConversion;
        ColorConversion prepareGrayConversion(const uchar* pImage, const std::size_t iHeight, const std::size_t iWidth);
        void iplRGB2GRAY(ColorConversion& eData, const Separable eCase = Separable::FALSE);
    } // namespace detail
} // namespace ImgProcLab
