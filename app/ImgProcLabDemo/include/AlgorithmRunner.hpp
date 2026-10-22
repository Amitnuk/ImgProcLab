#pragma once
#include "ArgumentParser.hpp"
#include "ImgProc/cuda/gaussian/GaussianCuda.hpp"
#include "ImgProc/cuda/gaussian/GaussianData.hpp"

using namespace ImgProcLab::detail;

class AlgorithmRunner
{
    public :
        AlgorithmRunner() = delete;
        ~AlgorithmRunner() = default;
        explicit AlgorithmRunner(const AlgorithmConfig& eConfig) :m_eConfig(eConfig) {}
        void run();


    private :
        void prepare();
        template<typename Function>
        void benchmark(Function&& function, const cv::Mat& oImage,const std::string& sAlgorithm="-",const std::string& sVersion="-",bool bVisualize = false );
        const AlgorithmConfig& m_eConfig;
        GaussianData m_eGaussianData;
       

};
