#include <optional> 
#include <string_view>

// ============================================================
// 1. Enumerations
// ============================================================

enum class Algorithm
{
    BoxFilter,
    GaussianFilter,
    Grayscale,
    Histogram,    
    MedianFilter,
    Resize,    
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
    std::optional<int> m_iHeight;
    std::optional<int> m_iWidth;    
    std::optional<int> m_iKernelSize;
    std::optional<float> m_fSigma;
    std::optional<std::string_view> m_sImagePath;
    std::optional<Algorithm> m_oAlgorithmType;
    std::optional<Version>   m_oVersion; 
};


class Parser
{
    public :
        Parser() = default;
        ~Parser() = default;
        std::optional<Algorithm> parseAlgorithm(std::string_view sFilterName);
        std::optional<Version> parseVersion(std::string_view sVersion);
        std::optional<int> parseInteger(std::string_view sInteger);
        std::optional<float> parseFloat(std::string_view sInteger);
        bool isCompatible(Algorithm oAlgorithmType, Version oVersion);
        void validateConfig(const AlgorithmConfig& oConfig);
        void executeAlgorithm(const AlgorithmConfig& oConfig);
        void printUsage(const char* executable);

        const AlgorithmConfig& parseArguments(int argc, char* argv[]);
       
        private :
            AlgorithmConfig m_oConfig;
};

