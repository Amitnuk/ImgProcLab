#pragma once 
#include <string>
#include "BenchmarkResult.hpp"

class BenchmarkWriter
{
private:
    /* data */
public:
    BenchmarkWriter() = delete;
    static void writeCSV(const std::string& sPath, const BenchmarResult& oBenchmarkResult);
    ~BenchmarkWriter();
};

