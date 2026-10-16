#include <iostream>
#include <memory>
#include "ArgumentParser.hpp"



int main(int argc, char* argv[]) 
{
    std::unique_ptr<Parser> pParser = std::make_unique<Parser>();

    if( !pParser )
    {
        std::cerr << "could not build parser" << std::endl;
        return 1;
    }
    
    if(argc == 2 && ( argv[1] == "--help" || argv[1] == "-h" ) )
    {
        pParser->printUsage(argv[0]);
        return 0;
    }

    
    try
    {
        const AlgorithmConfig& oConfig = pParser->parseArguments(argc, argv);
        pParser->executeFilter(oConfig);
	pParser->validateConfig(oConfig);        
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;

        pParser->printUsage(argv[0]);
        return 1;
    }
    
}
