#include <charconv>
#include <cstddef>
#include <exception>
#include <iostream>
#include "ArgumentParser.hpp"
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include "ImgProc/filters/GaussianFilter.hpp"

// ============================================================
// 3. String to enum conversion
// ============================================================


std::optional<Algorithm> Parser::parseAlgorithm(std::string_view sFilterName)
{
    if( sFilterName == "box") return Algorithm::BoxFilter;
    if( sFilterName == "gaussian") return Algorithm::GaussianFilter;
    if( sFilterName == "gray") return Algorithm::Grayscale;
    if( sFilterName == "histogram") return Algorithm::Histogram;    
    if( sFilterName == "median") return Algorithm::MedianFilter;
    if( sFilterName == "sobel") return Algorithm::SobelFilter;
    if( sFilterName == "resize") return Algorithm::Resize;
   
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
    if (sVersion == "all") return Version::ALL; 
    
    return std::nullopt;
}

// ============================================================
// 4. Float and Integer conversion
// ============================================================

std::optional<int> Parser::parseInteger(std::string_view sInteger)
{
    const char* pFirst = sInteger.data();
    const char* pLast   = pFirst + sInteger.size();
    int iVersion = 0;

    auto [ptr, ec] = std::from_chars(pFirst, pLast, iVersion); // structure binding
    return (ec == std::errc{} && ptr == pLast) ? std::optional<int>(iVersion) : std::nullopt;
}


std::optional<float>Parser::parseFloat(std::string_view sInteger)
{
  try
  {
    std::string sText(sInteger);
    std::size_t pos = 0;
    std::optional<float> result = std::stof(sText, &pos);

    if(pos != sText.size())
    {
      return std::nullopt;
    }

    return result;
  }
  catch(const std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return std::nullopt;
  }
  
}    
// ============================================================
// 5. Filter/version compatibility
// ============================================================

bool Parser::isCompatible(Algorithm oAlgorithmType, Version oVersion)
{

    if(oVersion == Version::ALL)
    {
      return true;
    }
    
    switch (oAlgorithmType)
    {
    case Algorithm::GaussianFilter:
      return oVersion >= Version::V1 && oVersion <= Version::V6; //V1-V6

    case Algorithm::BoxFilter:
    case Algorithm::Histogram:
    case Algorithm::Grayscale:      
    case Algorithm::MedianFilter:
    case Algorithm::SobelFilter:
    case Algorithm::Resize:      
      return (oVersion >= Version::V1) && (oVersion <= Version::V3); // only the gaussian blur have 6 version
    default:
        return false;
    }
   
}


// ============================================================
// 6. Configuration validation
// ============================================================
void Parser::validateConfig(const AlgorithmConfig& oConfig)
{
    if( !oConfig.m_sImagePath || (*oConfig.m_sImagePath).empty() )
    {
      throw std::invalid_argument("Missing required argument: --image");
    }

    if(!oConfig.m_eAlgorithmType)
    {
       throw std::invalid_argument("Missing required argument: --algo");
    }

    if(!oConfig.m_eVersion)
    {
       throw std::invalid_argument("Missing required argument: --version");
    }
    
    
    if(!isCompatible(*oConfig.m_eAlgorithmType, *oConfig.m_eVersion))
    {
       throw std::invalid_argument("Selected version is not supported by the selected algorithm.");
    }

    switch(*oConfig.m_eAlgorithmType)
    {
       case Algorithm::GaussianFilter:
        if(!oConfig.m_iKernelSize)
        {
        throw std::invalid_argument("Selected algorithim requires --kernel-size");
        }

        if(*oConfig.m_iKernelSize <= 0 || *oConfig.m_iKernelSize%2 == 0)
        {
            throw std::invalid_argument("Kernel size must be a positive odd integer");
        }
        
        if( !oConfig.m_fSigma ||  ( !std::isfinite(*oConfig.m_fSigma) || *oConfig.m_fSigma <= 0.0f ) )
        {
            throw std::invalid_argument("Gaussian sigma must be finite and positive");
        }

        if(oConfig.m_iHeight || oConfig.m_iWidth)
        {
        throw std::invalid_argument("Gaussian algorithm does not requires --width or --height");
        }
        break;
       case Algorithm::MedianFilter:
       case Algorithm::SobelFilter:
       case Algorithm::BoxFilter:         
        if(!oConfig.m_iKernelSize)
        {
            throw std::invalid_argument("Selected algothim requires --kernel-size");
        }

        if(*oConfig.m_iKernelSize <= 0 || *oConfig.m_iKernelSize%2 == 0)
        {
            throw std::invalid_argument("Kernel size must be a positive odd integer");
        }
        if( oConfig.m_fSigma || oConfig.m_iHeight || oConfig.m_iWidth)
        {
            throw std::invalid_argument("Unsupported parameters for Gaussian filter");
        }
        break;
       case Algorithm::Histogram:
       case Algorithm::Grayscale:
        if(oConfig.m_iKernelSize || oConfig.m_fSigma || oConfig.m_iHeight || oConfig.m_iWidth)
        {
        throw std::invalid_argument(
            "Selected algorithm does not requires"
        "--kernel-size, --sigma or --width or --height");
        }
        break;
       case Algorithm::Resize:
        if(!oConfig.m_iHeight || !oConfig.m_iWidth)
        {
	        throw std::invalid_argument("Resize requires --width and --height");
        }
         
        if(*oConfig.m_iHeight<=0 || *oConfig.m_iWidth <= 0)
        {
	        throw std::invalid_argument("Resize dimensions must be positive");
        }

	    if(oConfig.m_fSigma || oConfig.m_iKernelSize)
        {
	        throw std::invalid_argument("Resize does not requires sigma or sigma");
        }
	    break;
    }    
}


// ============================================================
// 7. Command-line argument parsing
// ============================================================

const AlgorithmConfig& Parser::parseArguments(int argc, char* argv[])
{
    std::optional<Algorithm> oAlgorithm;
    std::optional<Version> oVersion;
    std::optional<std::string_view> oImagePath;
    std::optional<int> oKernelSize;
    std::optional<float> oSigma;    
    std::optional<int> oWidth;
    std::optional<int> oHeight;
    std::optional<int> oIterations;
    std::optional<int> oWarmUps;
    int iIterations;
    int iGPUWarmUps;
    bool bBenchmark = false;
    bool bIsIterationProvided = false;
    bool bIsWarmupProvided = false;
    
    for(size_t i = 1; i < argc; i++)
    {
        std::string_view sArgument = argv[i];

        auto nextValue = [&]() -> std::string_view
        {
            if(i + 1 >= argc)
	        {
	            throw std::invalid_argument("Missing value for: "+std::string(sArgument));
            }
	        return argv[++i];
        };

        auto argumentCheck =[](const std::string& sArgument, const bool bState)
        {
            if(bState)
            {
                throw std::invalid_argument("Duplicate argument: "+std::string(sArgument));
            } 
        };

        if( sArgument == "--image")
        {
           if( oImagePath )
           {
                throw std::invalid_argument("Duplicate argument: --image");
           }

           oImagePath = nextValue();
           if(oImagePath->empty() || oImagePath->rfind("--", 0) == 0)
           {
                throw std::invalid_argument("Invalid image path");
	        }
        }
        else if( sArgument == "--algo")
        {
            /*if(oAlgorithm)
            {
                throw std::invalid_argument("Duplicate argument: --algo");
            }*/

            argumentCheck("--algo", static_cast<bool>(oAlgorithm));
            oAlgorithm = parseAlgorithm(nextValue());   
        }
        else if(sArgument == "--version")
        {
            if(oVersion)
            {
               throw std::invalid_argument("Duplicate argument: --version");
            }
            
            oVersion = parseVersion(nextValue());
        }
        else if(sArgument == "--kernel-size")
        {
            if(oKernelSize)
            {
                throw std::invalid_argument("Duplicate argument: --kernel-size");
            }
            oKernelSize = parseInteger(nextValue());
        }
        else if(sArgument == "--sigma")
        {
            if(oSigma)
	    {
	      throw std::invalid_argument("Duplicate argument: --sigma");
            }

	    oSigma = parseFloat(nextValue());
        }
        else if(sArgument == "--height")
        {
            if(oHeight)
            {
	            throw std::invalid_argument("Duplicate argument : --height");
	        }            

	        oHeight = parseInteger(nextValue());
        }
        else if(sArgument == "--width")
        {
            if(oWidth)
                {
            throw std::invalid_argument("Duplicate argument : --width");
            }            
	        oWidth = parseInteger(nextValue());
        }
        else if(sArgument == "--benchmark")
        {
            if(bBenchmark)
            {
                throw std::invalid_argument("Duplicate argument : --benchmark");
            }
            bBenchmark = true;    
        }
        else if(sArgument == "--iterations")
        {
            if (bIsIterationProvided)
            {
                throw std::invalid_argument("Duplicate argument : --iterations");
            }
            oIterations = parseInteger(nextValue());


            if( !oIterations || *oIterations <= 0)
            {
                throw std::invalid_argument("Number of iteration must be a positive number");
            }
            
            iIterations = *oIterations;
            bIsIterationProvided = true;
        }
        else if(sArgument == "--warmups")
        {
            if (bIsWarmupProvided)
            {
                throw std::invalid_argument("Duplicate argument : --warmups");
            }
            oWarmUps = parseInteger(nextValue());

            if( !oWarmUps || *oWarmUps <= 0  )
            {
                throw std::invalid_argument("Warmup must be a positive number");
            }
            



            iGPUWarmUps = *oWarmUps;
            bIsWarmupProvided = true;
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
    else if(oAlgorithm)
    {
        switch( *oAlgorithm )
	    {
            case Algorithm::BoxFilter:
            case Algorithm::GaussianFilter:
            case Algorithm::MedianFilter:
            case Algorithm::SobelFilter:
                if(!oKernelSize)
                {
                    throw std::invalid_argument("Selected algoritm requires argument: --kernel-size");
                }
                break;
            case Algorithm::Histogram:
            case Algorithm::Grayscale:            
            case Algorithm::Resize:            
                if(oKernelSize)
                {
                 throw std::invalid_argument("Selected algoritm does not requires argument: --kernel-size");
                }
	            break;
	    }
    }

    if(!oVersion)
    {
        throw std::invalid_argument("Missing required argument: --version");
    }

    if(!bBenchmark && (bIsWarmupProvided || bIsIterationProvided))
    {
        throw std::invalid_argument("--iterations and --warmaps requires --benchmark");
    }
    
    
    m_oConfig = AlgorithmConfig{ iIterations, iGPUWarmUps, bBenchmark, oHeight, oWidth, oKernelSize, oSigma, oImagePath, oAlgorithm, oVersion};
    
    validateConfig(m_oConfig);        

    return m_oConfig;
}





// ============================================================
// 8. Filter dispatcher
// ============================================================


void Parser::executeAlgorithm(const AlgorithmConfig& oConfig)
{
    
}


// ============================================================
// 9. Usage
// ============================================================

void Parser::printUsage(const char* executable)
{
     std::cout << "Usage:\n"
               << " " << executable << " --image <image>"
               << " --algo <name>"
               << " --version <version>"
               << " --kernel-size <odd_integer>"
               << " --sigma <float>"
               << " --height <int>"
               << " --width <int>\n\n"

               << "Available algorithms:\n"
               << "  box, gaussian, histogram, median, sobel\n\n"

               << "Available implementations:\n"
               << "  Gaussian: V1-V6\n"
               << "  Others:   V1-V3\n\n"

               << "Example:\n"
               << "  " << executable
               << " --image /path/to/image/folder/someimage.png"
               << " --algo gaussian"
               << " --version v2"
               << " --kernel-size 7"
               << " --sigma 0.7\n";      
}
