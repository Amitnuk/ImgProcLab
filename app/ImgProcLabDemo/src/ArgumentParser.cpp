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

// ============================================================
// 3. String to enum conversion
// ============================================================


std::optional<Algorithm> Parser::parseAlgorithm(std::string_view sFilterName)
{
    if( sFilterName == "box") return Algorithm::BoxFilter;
    if(sFilterName == "gaussian") return Algorithm::GaussianFilter;
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
    if(oConfig.m_sImagePath.empty())
    {
      throw std::invalid_argument("Missing required argument: --image");
    }
    
    if(!isCompatible(oConfig.m_oAlgorithmType, oConfig.m_oVersion))
    {
       throw std::invalid_argument("Selected version is not supported by the selected algorithm.");
    }

    switch(oConfig.m_oAlgorithmType)
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
	 
	 if( (oConfig.m_fSigma) && ( *oConfig.m_fSigma <= 0.0f || *oConfig.m_fSigma == std::numeric_limits<float>::infinity()) )
         {
	   throw std::invalid_argument("Gaussian sigma must be finite and positive");
         }

         if(*oConfig.m_iHeight || *oConfig.m_iWidth)
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
    std::optional<std::string> oImagePath;
    std::optional<int> oKernelSize;
    std::optional<float> oSigma;    
    std::optional<int> oWidth;
    std::optional<int> oHeight;
    
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

        if( sArgument == "--image")
        {
           if( oImagePath )
	   {
	     throw std::invalid_argument("Duplicate argument: --image");
           }

           oImagePath = std::string(nextValue());
	   
           if(oImagePath->empty() || oImagePath->rfind("--", 0) == 0)
           {
	     throw std::invalid_argument("Invalid image path");
	   }
        }
        else if( sArgument == "--algo")
        {
            if(oAlgorithm)
            {
                throw std::invalid_argument("Duplicate argument: --algo");
            }
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
        else
        {
            throw std::invalid_argument("Unknown argument: "  + std::string(sArgument));
        }        
    }


    if(!oAlgorithm)
    {
        throw std::invalid_argument("Missing required argument: --algo");
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

    
    
    m_oConfig = AlgorithmConfig{*oHeight, *oWidth, *oKernelSize, *oSigma, *oImagePath, *oAlgorithm, *oVersion};
    
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

    if(*oConfig.m_iKernelSize)
    {
       std::cout << "Kernel size: " << *oConfig.m_iKernelSize << std::endl;
    }

    if(*oConfig.m_iHeight)
    {
       std::cout << "Height: " << *oConfig.m_iHeight << std::endl;
    }

    if(*oConfig.m_iWidth)
    {
       std::cout << "Width: " << *oConfig.m_iWidth << std::endl;
    }

    if(*oConfig.m_fSigma > 0.0f)
    {
       std::cout << "Sigma: " << *oConfig.m_fSigma << std::endl;
    }
    
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
