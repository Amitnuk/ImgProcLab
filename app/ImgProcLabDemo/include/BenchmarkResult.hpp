#pragma once 


#include <string>



struct BenchmarResult
{
    int m_iWidth=0;
    int m_iHeight=0;
    int m_iKernelSize=0;
    int m_iWarmups=0;
    int m_iIterations=0;
    float m_faverageMs=0.0f;
    std::string m_sAlgorithm="-"; 
    std::string m_sVersion="-";
};
