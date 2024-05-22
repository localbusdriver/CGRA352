#include "main.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main(int argc, char **argv)
{
    srand((unsigned)time(0)); // Seed for random number generation

    if (argc > 1 && std::string(argv[1]) == "fem")
    {
        fem::FeatureExtractionAndMatching CORE;
        CORE.run();
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " <desired function>\n or leave blank to run all" << std::endl;
        fem::FeatureExtractionAndMatching CORE;
        CORE.run();
    }

    return 0;
}