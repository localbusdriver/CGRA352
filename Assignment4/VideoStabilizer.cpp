#include "main.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>

namespace vs
{
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
    }

    /** NOTE:
     * Should result in homographies BETWEEN consecutive frames
     * (i.e. homographies[0] is between frame 0 and frame 1, homographies[1] is between frame 1 and frame 2, etc.)
     * Therefore, if there are N images, there should be N-1 homographies
     */
    void VideoStabilizer::findHomography()
    {
        std::cout << "[INFO] Finding Homographies..." << std::endl;

        // Initialize
        std::vector<cv::KeyPoint> prevKeypoints, currKeypoints;
        cv::Mat prevDescriptors, currDescriptors;
        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");

        // Iteratively find homographies
        for (size_t i = 1; i < images.size(); ++i) // Start from the second image because we need to compare with the previous one
        {
            detector->detectAndCompute(images[i - 1], cv::Mat(), prevKeypoints, prevDescriptors); // Prev image
            detector->detectAndCompute(images[i], cv::Mat(), currKeypoints, currDescriptors);     // Curr image

            std::vector<cv::DMatch> matches;
            matcher->match(prevDescriptors, currDescriptors, matches);

            // Estimate homography
            std::vector<cv::Point2f> pts1, pts2;
            for (auto &match : matches)
            {
                pts1.push_back(prevKeypoints[match.queryIdx].pt);
                pts2.push_back(currKeypoints[match.trainIdx].pt);
            }

            cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC);
            homographies.push_back(H);
        }
    }

    // Function to compute Gaussian weights
    std::vector<double> VideoStabilizer::computeGaussianWeights()
    {
        std::cout << "[INFO] Computing Gaussian Weights..." << std::endl;
        std::vector<double> kernel(kernelSize);
        double sum = 0.0;
        const int halfSize = kernelSize / 2;

        for (int i = -halfSize; i <= halfSize; i++)
        {
            double value = exp(-0.5 * i * i / (sigma * sigma));
            kernel[i + halfSize] = value;
            sum += value;
        }

        // Normalize the kernel
        for (double &weight : kernel)
        {
            weight /= sum;
        }

        return kernel;
    }

    void VideoStabilizer::smoothHomographies()
    {
        std::cout << "[INFO] Smoothing Homographies..." << std::endl;

        // Initialize
        std::vector<double> weights = computeGaussianWeights();
        const int halfSize = kernelSize / 2;
        size_t numHomographies = homographies.size();
        std::vector<cv::Mat> smoothedHomographies(numHomographies);

        for (int i = 0; i < numHomographies; i++)
        {
            cv::Mat smoothed = cv::Mat::zeros(3, 3, CV_64F); // Initialize smoothed homography
            for (int k = -halfSize; k <= halfSize; k++)      // Convolution
            {
                int idx = i + k;                       // Index of homography to be convolved
                if (idx >= 0 && idx < numHomographies) // Check if index is within bounds
                {
                    smoothed += weights[k + halfSize] * homographies[idx]; // Convolve
                }
            }
            smoothedHomographies[i] = smoothed; // Store smoothed homography
        }

        // Update homographies
        homographies = smoothedHomographies;
    }

    void VideoStabilizer::stabilize()
    {
        std::cout << "[INFO] Stabilizing Images..." << std::endl;

        cv::Mat cumulativeHomography = cv::Mat::eye(3, 3, CV_64F); // Start with an identity matrix for the first frame

        for (size_t i = 0; i < images.size() - 1; i++) // For all images
        {
            cumulativeHomography *= homographies[i]; // Update the cumulative transformation w/ corresponding homography
            cv::Mat stabilizedImage;
            cv::warpPerspective(images[i], stabilizedImage, cumulativeHomography, images[i].size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT); // Warp image
            stabilizedImages.push_back(stabilizedImage);
        }
    }

    void VideoStabilizer::saveShow()
    {
        for (size_t i = 0; i < stabilizedImages.size(); i++)
        {
            cv::imshow("Stabilized Image", stabilizedImages[i]);
            cv::waitKey(100); // Display each frame for 100 ms
            cv::imwrite("./COMPLETION/stabilized_frame" + std::to_string(i) + ".png", stabilizedImages[i]);
        }
    }

    void VideoStabilizer::run()
    {
        findHomography();
        smoothHomographies();
        stabilize();
        saveShow();
    }
}