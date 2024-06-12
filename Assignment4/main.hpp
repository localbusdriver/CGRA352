#ifndef MAIN_H
#define MAIN_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace fem
{
    /**
     * A class for performing feature extraction and matching between two images
     * using SIFT descriptors and RANSAC for homography estimation.
     */
    class FeatureExtractionAndMatching
    {
    public:
        FeatureExtractionAndMatching(std::string inputDir);
        void run();

    private:
        void SIFT();
        void match();
        void RANSAC();
        cv::Mat warpImage();
        cv::Mat drawPart1();
        cv::Mat drawPart2();
        void show();

        // Images
        cv::Mat frame39, frame41;

        // KeyPoints and Descriptors
        cv::Mat descriptors1, descriptors2;
        std::vector<cv::KeyPoint> keypoints1, keypoints2;

        std::vector<cv::DMatch> matches;

        // RANSAC params
        cv::Mat bestHomography;
        int bestInliers = 0;
        const int iterations = 1000;
        const float reprojectionThreshold = 3.0f; // Distance in pixels to consider an inlier
    };
}

namespace vs
{
    class VideoStabilizer
    {
    public:
        VideoStabilizer(std::string inputDir);
        void run();

    private:
        void loadImages(std::string inputDir);
        void findHomography();
        void computeCumulativeHomographies();
        std::vector<double> computeGaussianWeights();
        void smoothHomographies();
        void computeStabilizationTransforms();
        void stabilize();
        void saveShow();
        void saveHomographies();
        void createDir();

        const int kernelSize = 5;
        const double sigma = 1.0;

        std::vector<cv::Mat> images;
        std::vector<cv::Mat> homographies;
        std::vector<cv::Mat> cumulativeHomographies;
        std::vector<cv::Mat> smoothedHomographies;
        std::vector<cv::Mat> stabilizationTransforms;
        std::vector<cv::Mat> stabilizedImages;
    };
}

namespace cw
{
    class CroppingWindow
    {
    public:
        CroppingWindow();
        void run();

    private:
        void transformMask();
        // cv::Mat transformMask(cv::Mat& mask, const cv::Mat& transformation);
        cv::Mat generateCombinedMask();
        cv::Rect findLargestInscribedSquare(const cv::Mat& mask);
        cv::Rect findLargestInscribedRectangle(const cv::Mat& combinedMask);
        cv::Rect adjustToAspectRatio(cv::Rect largetSquare);
        void cropAndSaveFrames(const cv::Rect& cropRect);
        void loadImages();
        void createDir();
        void show();

        std::vector<cv::Mat> masks;
        std::vector<cv::Mat> frames;
        std::vector<cv::Mat> homographies;
    };
}
#endif // MAIN_H