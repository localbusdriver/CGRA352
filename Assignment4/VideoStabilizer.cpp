#include "main.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace vs
{

    /** NOTE:
     * Should result in homographies BETWEEN consecutive frames
     * (i.e. homographies[0] is between frame 0 and frame 1, homographies[1] is between frame 1 and frame 2, etc.)
     * Therefore, if there are N images, there should be N-1 homographies
     */
    void VideoStabilizer::findHomography()
    {
        std::cout << "[INFO] Finding Homographies..." << std::endl;

        // Initialize
        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();

        // Iteratively find homographies
        for (size_t i = 1; i < images.size(); ++i) // Start from the second image because we need to compare with the previous one
        {
            std::vector<cv::KeyPoint> kp1, kp2;
            cv::Mat desc1, desc2;

            detector->detectAndCompute(images[i - 1], cv::Mat(), kp1, desc1); // Prev image
            detector->detectAndCompute(images[i], cv::Mat(), kp2, desc2);     // Curr image

            cv::BFMatcher matcher(cv::NORM_L2);   // Brute force matcher
            std::vector<cv::DMatch> matches;      // Matches
            matcher.match(desc1, desc2, matches); // Match descriptors

            // Estimate homography
            std::vector<cv::Point2f> pts1, pts2;
            for (auto &match : matches)
            {
                pts1.push_back(kp1[match.queryIdx].pt); // Previous image
                pts2.push_back(kp2[match.trainIdx].pt); // Current image
            }

            cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC);
            homographies.push_back(H);
        }
    }

    void VideoStabilizer::computeCumulativeHomographies()
    {
        cv::Mat H_cumulative = cv::Mat::eye(3, 3, CV_64F);
        // cumulativeHomographies.push_back(H_cumulative.clone());
        for (const cv::Mat &H : homographies)
        {
            H_cumulative = H_cumulative * H;
            cumulativeHomographies.push_back(H_cumulative.clone());
        }
    }

    // Function to compute Gaussian weights
    std::vector<double> VideoStabilizer::computeGaussianWeights()
    {
        std::cout << "[INFO] Computing Gaussian Weights..." << std::endl;
        std::vector<double> weights(kernelSize);
        double sum = 0.0;
        const int halfSize = kernelSize / 2;

        for (int i = -halfSize; i <= halfSize; ++i)
        {
            double value = exp(-(i * i) / (2 * sigma * sigma));
            weights[i + halfSize] = value;
            sum += value;
        }

        // Normalize the kernel
        for (double &weight : weights)
        {
            weight /= sum;
        }

        return weights;
    }

    void VideoStabilizer::smoothHomographies()
    {
        std::cout << "[INFO] Smoothing Homographies..." << std::endl;

        // Initialize
        std::vector<double> weights = computeGaussianWeights();
        const int halfSize = kernelSize / 2;
        size_t numHomographies = cumulativeHomographies.size();
        smoothedHomographies = cumulativeHomographies;

        for (size_t i = halfSize; i < numHomographies - halfSize; ++i)
        {
            cv::Mat smoothed = cv::Mat::zeros(3, 3, CV_64F);
            for (int j = -halfSize; j <= halfSize; ++j)
            {
                smoothed += weights[j + halfSize] * cumulativeHomographies[i + j];
            }
            smoothedHomographies[i] = smoothed;
        }
    }

    void VideoStabilizer::computeStabilizationTransforms()
    {
        for (size_t i = 0; i < cumulativeHomographies.size(); ++i)
        {
            cv::Mat smoothed_inv;
            invert(smoothedHomographies[i], smoothed_inv);
            cv::Mat U = smoothed_inv * cumulativeHomographies[i];
            stabilizationTransforms.push_back(U);
        }
    }

    void VideoStabilizer::stabilize()
    {
        std::cout << "[INFO] Stabilizing Images..." << std::endl;

        // cv::Mat cumulativeHomography = cv::Mat::eye(3, 3, CV_64F); // Start with an identity matrix for the first frame

        for (size_t i = 0; i < images.size() - 1; i++) // For all images
        {
            // cumulativeHomography *= homographies[i]; // Update the cumulative transformation w/ corresponding homography
            cv::Mat stabilizedImage;
            cv::warpPerspective(images[i], stabilizedImage, stabilizationTransforms[i], images[i].size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT); // Warp image
            stabilizedImages.push_back(stabilizedImage);
        }
    }

    void VideoStabilizer::saveShow()
    {
        for (size_t i = 0; i < stabilizedImages.size(); i++)
        {
            cv::imshow("Stabilized Image", stabilizedImages[i]);
            cv::waitKey(100); // Display each frame for 100 ms
            std::string num = std::to_string(i);
            num = std::string(3 - num.length(), '0') + num; // e.g. if index is 3; num = "003"
            cv::imwrite("./COMPLETION/stable" + num + ".png", stabilizedImages[i]);
        }
    }

    void VideoStabilizer::saveHomographies()
    {
        int i = 0;
        for (const auto &smoothed : stabilizationTransforms)
        {
            // Save smoothed homography to file (XML format
            std::string num = std::to_string(i);
            num = std::string(3 - num.length(), '0') + num; // e.g. if index is 3; num = "003"
            std::string filename = "./HOMOGRAPHIES/homography" + num + ".xml";
            cv::FileStorage fs(filename, cv::FileStorage::WRITE);
            fs << "homography" << smoothed;
            fs.release();
            i++;
        }
    }

    VideoStabilizer::VideoStabilizer(std::string inputDir)
    {
        std::cout << "\n[INFO] Initializing Video Stabilizer..." << std::endl;
        loadImages(inputDir);
    }

    void VideoStabilizer::loadImages(std::string inputDir)
    {
        std::cout << "[INFO] Loading Images..." << std::endl;
        for (int i = 0; i < 102; ++i)
        {
            std::string num = std::to_string(i);
            num = std::string(3 - num.length(), '0') + num; // e.g. if index is 3; num = "003"

            std::string filename = inputDir + "/Frame" + num + ".jpg";
            cv::Mat img = cv::imread(filename);
            if (img.empty())
                break;
            images.push_back(img);
        }
        std::cout << "[INFO] Completed Loading Images..." << std::endl;
    }

    void VideoStabilizer::run()
    {
        createDir();
        findHomography();
        computeCumulativeHomographies();
        smoothHomographies();
        computeStabilizationTransforms();
        stabilize();
        saveShow();
        saveHomographies();
    }

    void VideoStabilizer::createDir()
    {
        fs::path dir = "./COMPLETION";
        fs::path dir2 = "./HOMOGRAPHIES";
        try
        {
            if (fs::create_directory(dir))
                std::cout << "[INFO] Directory Created: " << dir << std::endl;
            else
                std::cout << "[INFO] Directory Already Exists: " << dir << std::endl;
            
            if (fs::create_directory(dir2))
            {
                std::cout << "[INFO] Directory Created: " << dir2 << std::endl;
            }
            else
            {
                std::cout << "[INFO] Directory Already Exists: " << dir2 << std::endl;
            }
        }
        catch (const fs::filesystem_error &e)
        {
            std::cerr << "Error creating directory: " << e.what() << '\n';
        }

        std::cout << "[INFO] Created 'COMPLETION' and 'HOMOGRAPHIES' directory" << std::endl;
    }
}