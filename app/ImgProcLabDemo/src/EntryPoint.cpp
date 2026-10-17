#include <iostream>
#include <memory>
#include "ArgumentParser.hpp"



int main(int argc, char* argv[]) 
{
    Parser oParser;

    if(argc == 2 && ( std::string_view{argv[1]} == "--help" || std::string_view{argv[1]} == "-h" ) )
    {
        oParser.printUsage(argv[0]);
        return 0;
    }

    
    try
    {
        const AlgorithmConfig& oConfig = oParser.parseArguments(argc, argv);
        oParser.executeAlgorithm(oConfig);
	    
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;

        oParser.printUsage(argv[0]);
        return 1;
    }
    
}
