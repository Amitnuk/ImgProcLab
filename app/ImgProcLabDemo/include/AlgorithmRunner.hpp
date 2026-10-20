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
        void execute();
        const AlgorithmConfig& m_eConfig;
        GaussianData m_eGaussianData;

};