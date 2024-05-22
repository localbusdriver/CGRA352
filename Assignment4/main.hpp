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
        FeatureExtractionAndMatching();
        void run();

    private:
        void SIFT();
        void match();
        void RANSAC();
        void show();
        cv::Mat drawPart1();
        cv::Mat drawPart2();

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
#endif // MAIN_H