#include <optional> 
#include <string_view>

// ============================================================
// 1. Enumerations
// ============================================================

enum class Algorithm
{   
    BoxFilter,
    Histogram,
    GaussianFilter,
    MedianFilter,
    SobelFilter,
};


enum class Version
{
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    ALL
};

// ============================================================
// 2. Configuration
// ============================================================ 

struct AlgorithmConfig
{
    Algorithm m_oAlgorithmType;
    Version   m_oVersion;
    int m_iKernelSize;
};


class Parser
{
    public :
        Parser() = default;
        ~Parser() = default;
        std::optional<Algorithm> parseAlgorithm(std::string_view sFilterName);
        std::optional<Version> parseVersion(std::string_view sVersion);
        std::optional<int> parseInteger(std::string_view sInteger);
        bool isCompatible(Algorithm oAlgorithmType, Version oVersion);
        void validateConfig(const AlgorithmConfig& oConfig);
        void executeFilter(const AlgorithmConfig& oConfig);
        void printUsage(const char* executable);

        const AlgorithmConfig& parseArguments(int argc, char* argv[]);
       
        private :
            AlgorithmConfig m_oConfig;
};

