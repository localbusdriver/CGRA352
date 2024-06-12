#include "main.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

void printUsage(char *argv[])
{
    std::cout << "\n Usage: " << argv[0] << "  <desired function> <relative path to input images>(e.g. core ../input)\n or leave blank to run all." << std::endl;
    std::cout << "\nPlease include the relative path to the input images as the second argument.\nIf not mentioned, it will default to '../input'." << std::endl;
    std::cout << "\nValid arguments:" << std::endl;
    std::cout << "'core'\tTo run CORE implementations\n'comp' || 'completion'\tTo run COMPLETION implementation\n'chal' || 'challlenge'\tTo run CHALLENGE implementation\n<blank>\tTo run all functions\n"
              << std::endl;
}

int main(int argc, char **argv)
{
    srand((unsigned)time(0)); // Seed for random number generation

    // Creates output directory for COMPLETION
    std::string outputDir = "/COMPLETION";
    const char *path_c = outputDir.c_str();
    int stat = mkdir(path_c, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    std::cout << "\nPlease note that this code will create an output directory for the results of COMPLETION" << std::endl;

    std::string inputDir = "../input";
    if (argc > 1 && argv[argc - 1] != NULL)
    {
        if ((std::string(argv[1]) == "core" || std::string(argv[1]) == "comp" || std::string(argv[1]) == "completion" || std::string(argv[1]) == "chal" || std::string(argv[1]) == "challenge"))
            inputDir = argc > 2 ? argv[argc - 1] : inputDir;
        else if (std::string(argv[1]) == "view")
            inputDir = inputDir;
        else
            inputDir = argv[argc - 1];
    }

    if (argc > 1 && std::string(argv[1]) == "core")
    {
        fem::FeatureExtractionAndMatching CORE(inputDir = inputDir); // CORE Part 1/2/3
        CORE.run();
    }
    else if (argc > 1 && (std::string(argv[1]) == "comp" || std::string(argv[1]) == "completion"))
    {
        std::cout << "\n[INFO] Running COMPLETION..." << std::endl;
        vs::VideoStabilizer VS(inputDir = inputDir); // COMPLETION
        VS.run();
    }
    else if (argc > 1 && (std::string(argv[1]) == "chal" || std::string(argv[1]) == "challenge"))
    {
        cw::CroppingWindow CHAL; // CHALLENGE
        CHAL.run();
    }
    else
    {
        printUsage(argv);
        fem::FeatureExtractionAndMatching CORE(inputDir = inputDir);
        CORE.run();
        vs::VideoStabilizer VS(inputDir = inputDir); // COMPLETION
        VS.run();
        cw::CroppingWindow CHAL; // CHALLENGE
        CHAL.run();
    }

    return 0;
}