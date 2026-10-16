#include <charconv>
#include <iostream>
#include "ArgumentParser.hpp"
#include <stdexcept>

// ============================================================
// 3. String to enum conversion
// ============================================================


std::optional<Algorithm> Parser::parseAlgorithm(std::string_view sFilterName)
{
    if( sFilterName == "box") return Algorithm::BoxFilter;
    if( sFilterName == "gaussian") return Algorithm::GaussianFilter;
    if( sFilterName == "median") return Algorithm::MedianFilter;
    if( sFilterName == "sobel") return Algorithm::SobelFilter;

    return std::nullopt;
}



std::optional<Version> Parser::parseVersion(std::string_view sVersion)
{
    if (sVersion == "v1") return Version::V1; 
    if (sVersion == "v2") return Version::V2; 
    if (sVersion == "v3") return Version::V3; 
    if (sVersion == "v4") return Version::V4; 
    if (sVersion == "v5") return Version::V5; 
    if (sVersion == "v6") return Version::V6; 
    
    return std::nullopt;
}

// ============================================================
// 4. Integer conversion
// ============================================================

std::optional<int> Parser::parseInteger(std::string_view sInteger)
{
    const char* pFirst = sInteger.data();
    const char* pLast   = pFirst + sInteger.size();
    int iVersion = 0;

    auto [ptr, ec] = std::from_chars(pFirst, pLast, iVersion); // structure binding
    return (ec == std::errc{} && ptr == pLast) ? std::optional<int>(iVersion) : std::nullopt;
}

// ============================================================
// 5. Filter/version compatibility
// ============================================================

bool Parser::isCompatible(Algorithm oAlgorithmType, Version oVersion)
{
    switch (oAlgorithmType)
    {
    case Algorithm::GaussianFilter:
        return true; //V1-V6

    case Algorithm::BoxFilter:
    case Algorithm::Histogram:
    case Algorithm::MedianFilter:
    case Algorithm::SobelFilter: 
        return ( oVersion == Version::V1 ) || ( oVersion == Version::V2 ) || (oVersion == Version::V3 ); // only the gaussian blur have 6 version
    
    default:
        return false;
    }

}


// ============================================================
// 6. Configuration validation
// ============================================================
void Parser::validateConfig(const AlgorithmConfig& oConfig)
{
    if(!isCompatible(oConfig.m_oAlgorithmType, oConfig.m_oVersion))
    {
       throw std::invalid_argument("Selected version is not supported by the selected filter.");
    }
    
    bool bShouldHaveKernel = oConfig.m_oAlgorithmType != Algorithm::Histogram;
    if( ( oConfig.m_iKernelSize <= 0 || oConfig.m_iKernelSize%2 == 0) && bShouldHaveKernel )
    {
        if(oConfig.m_iKernelSize ==  1)
        {
            std::cout << "[WARNING]: kernel size equal 1 equivalent to convolving with dirac function" << std::endl;
        }
        throw std::invalid_argument("the kernel size must be a positive odd integer ");
    }
    
    
    
}


// ============================================================
// 7. Command-line argument parsing
// ============================================================

const AlgorithmConfig& Parser::parseArguments(int argc, char* argv[])
{
    std::optional<Algorithm> oAlgorithm;
    std::optional<Version> oVersion;
    std::optional<int> oKernelSize;
    

    for(size_t i = 2; i < argc; i++)
    {
        std::string_view sArgument = argv[i];

        if( sArgument == "--algo")
        {
            if(oAlgorithm)
            {
                throw std::invalid_argument("Duplicate argument: --algo");
            }

            if(++i > argc)
            {
                throw std::invalid_argument("Missing value for --algo");
            }
            
            oAlgorithm = parseAlgorithm(argv[i]);

            if(!oAlgorithm)
            {
                throw std::invalid_argument("Invalid algorithm: " + std::string(argv[i]));
            }
            
        }
        else if(sArgument == "--version")
        {
            if(oVersion)
            {
               throw std::invalid_argument("Duplicate argument: --version");
            }

            if(++i > argc )
            {
                throw std::invalid_argument("Missing value for --algo");
            }
            
            oVersion = parseVersion(argv[i]);

            if(!oVersion)
            {
                throw std::invalid_argument("Invalid Version: " + std::string(argv[i]));
            }
            
            
            
        }
        else if(sArgument == "--kernel-size")
        {
            if(oKernelSize)
            {
                throw std::invalid_argument("Duplicate argument: --kernel-size");
            }

            if(++i > argc )
            {
                throw std::invalid_argument("Missing value for --kernel-size");
            }
            
            oKernelSize = parseInteger(argv[i]);

            if(!oKernelSize)
            {
                throw std::invalid_argument("Invalid kernel size: " + std::string(argv[i]));
            }
            
        }
        else
        {
            throw std::invalid_argument("Unknown argument: "  + std::string(sArgument));
        }
        
    }


    if(!oAlgorithm)
    {
        throw std::invalid_argument("Missing required argument: --algo");
    }

    if(!oVersion)
    {
        throw std::invalid_argument("Missing required argument: --version");
    }

    if(!oKernelSize)
    {
        throw std::invalid_argument("Missing required argument: --kernel-size");
    }
    
    m_oConfig = AlgorithmConfig{*oAlgorithm,*oVersion,*oKernelSize};
    validateConfig(m_oConfig);

    return m_oConfig;
}





// ============================================================
// 8. Filter dispatcher
// ============================================================


void Parser::executeFilter(const AlgorithmConfig& oConfig)
{
    switch(oConfig.m_oAlgorithmType)
    {
        case Algorithm::BoxFilter:
            switch(oConfig.m_oVersion)
            {
            case Version::V1:
                std::cout << "Executing BoxFilter V1" << std::endl;
                break;

            case Version::V2:
                std::cout << "Executing BoxFilter V2" << std::endl;
                break;

            case Version::V3:
                std::cout << "Executing BoxFilter V3" << std::endl;
                break;
            
            default:
                std::cout << "[ERROR]: Executing BoxFilter with Unknown Version" << std::endl;
                break;
            }
            break;
        
        case Algorithm::GaussianFilter:
            switch(oConfig.m_oVersion)
            {
            case Version::V1:
                std::cout << "Executing GaussianFilter V1" << std::endl;
                break;

            case Version::V2:
                std::cout << "Executing GaussianFilter V2" << std::endl;
                break;

            case Version::V3:
                std::cout << "Executing GaussianFilter V3" << std::endl;
                break;
            case Version::V4:
                std::cout << "Executing GaussianFilter V4" << std::endl;
                break;
            case Version::V5:
                std::cout << "Executing GaussianFilter V5" << std::endl;
                break;
            case Version::V6:
                std::cout << "Executing GaussianFilter V6" << std::endl;
                break;
            
            default:
                std::cout << "[ERROR]: Executing GaussianFilter with Unknown Version" << std::endl;
                break;
            }
            break;

        case Algorithm::Histogram :
            switch(oConfig.m_oVersion)
            {
            case Version::V1:
                std::cout << "Executing Histogram V1" << std::endl;
                break;

            case Version::V2:
                std::cout << "Executing Histogram V2" << std::endl;
                break;

            case Version::V3:
                std::cout << "Executing Histogram V3" << std::endl;
                break;
            
            default:
                std::cout << "[ERROR]: Executing Histogram with Unknown Version" << std::endl;
                break;
            }
            break;

        case Algorithm::MedianFilter :
            switch(oConfig.m_oVersion)
            {
            case Version::V1:
                std::cout << "Executing MedianFilter V1" << std::endl;
                break;

            case Version::V2:
                std::cout << "Executing MedianFilter V2" << std::endl;
                break;

            case Version::V3:
                std::cout << "Executing MedianFilter V3" << std::endl;
                break;
            
            default:
                std::cout << "[ERROR]: Executing MedianFilter with Unknown Version" << std::endl;
                break;
            }

        case Algorithm::SobelFilter :
            switch(oConfig.m_oVersion)
            {
            case Version::V1:
                std::cout << "Executing SobelFilter V1" << std::endl;
                break;

            case Version::V2:
                std::cout << "Executing SobelFilter V2" << std::endl;
                break;

            case Version::V3:
                std::cout << "Executing SobelFilter V3" << std::endl;
                break;
            
            default:
                std::cout << "[ERROR]: Executing SobelFilter with Unknown Version" << std::endl;
                break;
            }
        
        default:
            std::cout << "[ERROR]: Executing Unknown Algorithm" << std::endl;
            break;
    }

     std::cout << "Kernel size: " << oConfig.m_iKernelSize << std::endl;
}


// ============================================================
// 9. Usage
// ============================================================

void Parser::printUsage(const char* executable)
{
    std::cout << "Usage:\n"
              << " " << executable 
              << " image <image>"
              << " --algo <name>"
              << " --version <version>"
              << " --kernel-size <odd_integer>\n\n" 

              << "Available algorithms:\n"
              << "  box, gaussian, histogram, median, sobel\n\n"

              << "Available implementations:\n"
              << "  Gaussian: V1-V6\n"
              << "  Others:   V1-V3\n\n"

             << "Example:\n"
             << "  " << executable
             << "  /path/to/image/folder/someimage.png"
             << " --algo gaussian"
             << " --version v2"
             << " --kernel-size 7\n";
}