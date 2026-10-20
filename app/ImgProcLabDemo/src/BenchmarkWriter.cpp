#include <fstream>
#include <stdexcept>
#include "BenchmarkWriter.hpp"
#include <iostream>

void BenchmarkWriter::writeCSV(const std::string& sPath, const BenchmarResult& oBenchmarkResult)
{
    std::ofstream oFile(sPath);


    if(!oFile.is_open())
    {
        throw std::invalid_argument("Could not open file: "+ sPath);
    }

    oFile << "Algorithm,Version,Warmups,Iterations,KernelSize,Width,Height,AverageMs" << std::endl;
    oFile << oBenchmarkResult.m_sAlgorithm  << ',';
    oFile << oBenchmarkResult.m_sVersion  << ',';
    oFile << oBenchmarkResult.m_iWarmups  << ',';
    oFile << oBenchmarkResult.m_iIterations  << ',';
    oFile << oBenchmarkResult.m_iKernelSize  << ',';
    oFile << oBenchmarkResult.m_iWidth  << ',';
    oFile << oBenchmarkResult.m_iHeight  << ',';
    oFile << oBenchmarkResult.m_faverageMs << '\n';


    if(!oFile)
    {
        throw std::runtime_error("Could not write benchmark CSV: " + sPath);
    }

    std::cout << "Benchmark saved in: " << sPath << std::endl;
    
}