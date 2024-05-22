#include "main.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <random>

namespace fem
{
    FeatureExtractionAndMatching::FeatureExtractionAndMatching()
    { // Initialize SIFT, keypoints, descriptors, and matches && Find best matches
        std::cout << "[INFO] Loading Images..." << std::endl;
        frame39 = cv::imread("../inp/Frame039.jpg");
        frame41 = cv::imread("../inp/Frame041.jpg");

        if (frame39.empty() || frame41.empty())
        {
            std::cerr << "[Error] Image not found" << std::endl;
            exit(1);
        }
    }

    void FeatureExtractionAndMatching::SIFT()
    {
        std::cout << "[INFO] Performing SIFT..." << std::endl;
        // Create SIFT detector
        auto sift = cv::SIFT::create();

        // Detect keypoints and compute descriptors
        sift->detectAndCompute(frame39, cv::noArray(), keypoints1, descriptors1); // Detect keypoints and compute descriptors
        sift->detectAndCompute(frame41, cv::noArray(), keypoints2, descriptors2);
    }

    void FeatureExtractionAndMatching::match()
    {
        std::cout << "[INFO] Matching..." << std::endl;
        // Find best matches by trying each descriptor
        cv::BFMatcher matcher(cv::NORM_L2, true);           // Brute-force matcher
        matcher.match(descriptors1, descriptors2, matches); // Match descriptors
        if (matches.size() < 4)
        {
            std::cerr << "[Error] Not enough matches found" << std::endl;
            exit(1);
        }
    }

    void FeatureExtractionAndMatching::RANSAC()
    {
        std::cout << "[INFO] Performing RANSAC" << std::endl;

        std::mt19937 rng((unsigned)time(0));
        std::uniform_int_distribution<> dist(0, matches.size() - 1);
        const int sample_size = 4;

        for (int i = 0; i < iterations; i++)
        {
            // Randomly select 4 matches
            std::vector<cv::Point2f> srcPoints, dstPoints;
            for (int j = 0; j < 4; j++)
            {
                // int idx = rand() % matches.size();
                int idx = dist(rng);
                srcPoints.push_back(keypoints1[matches[idx].queryIdx].pt);
                dstPoints.push_back(keypoints2[matches[idx].trainIdx].pt);
            }

            // Estimate homography from these points
            cv::Mat H = cv::findHomography(srcPoints, dstPoints, 0);

            if (H.empty())
                continue; // Skip if homography estimation failed

            int inliers = 0;

            /**
             * For each pair of points (𝑝𝑖, 𝑞𝑖) in the set of matches,
             * compute the projection of 𝑝𝑖 using 𝐻 and compare it with 𝑞𝑖.
             */
            for (const auto &match : matches) // Compute inliers amongst all pairs
            {
                cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
                cv::Point2f pt2 = keypoints2[match.trainIdx].pt;

                cv::Mat ptHomogeneous = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);                     // Homogeneous coordinates
                cv::Mat ptTransformed = H * ptHomogeneous;                                                 // Apply homography
                double w = ptTransformed.at<double>(2);                                                    // Normalization factor
                cv::Point2f ptProjected(ptTransformed.at<double>(0) / w, ptTransformed.at<double>(1) / w); // Dehomogenize

                if (cv::norm(ptProjected - pt2) < reprojectionThreshold) // (|𝑝𝑖 − 𝐻𝑞𝑖| < 𝜀)
                    inliers++;
            }

            if (inliers > bestInliers)
            {
                bestInliers = inliers;      // Inlier pairs
                bestHomography = H.clone(); // best homography trasnform "H"
            }
        }
    }

    /**
     * Draw the matches between the two images for part 1;
     * SIFT feature extraction and matching.
     */
    cv::Mat FeatureExtractionAndMatching::drawPart1()
    {
        std::cout << "[INFO] Drawing part 1 image" << std::endl;
        // Draw Image
        cv::Mat res(frame39.rows + frame41.rows, std::max(frame39.cols, frame41.cols), frame39.type());
        cv::Mat upper(res, cv::Rect(0, 0, frame39.cols, frame39.rows));
        cv::Mat lower(res, cv::Rect(0, frame39.rows, frame41.cols, frame41.rows));
        frame39.copyTo(upper);
        frame41.copyTo(lower);

        // Draw green lines for matches
        for (const auto &match : matches)
        {
            cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
            cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
            pt2.y += frame39.rows; // Adjust y-coordinate for the second image

            cv::line(res, pt1, pt2, cv::Scalar(0, 255, 0)); // BGR for green
        }
        return res;
    }

    /**
     * Draw the inliers and outliers between the two images for part 2;
     * Estimate Homography transformation (using RANSAC).
     */
    cv::Mat FeatureExtractionAndMatching::drawPart2()
    {
        std::cout << "[INFO] Drawing part 2 image" << std::endl;
        cv::Mat res(frame39.rows + frame41.rows, std::max(frame39.cols, frame41.cols), frame39.type());
        frame39.copyTo(res(cv::Rect(0, 0, frame39.cols, frame39.rows)));
        frame41.copyTo(res(cv::Rect(0, frame39.rows, frame41.cols, frame41.rows)));

        for (const auto &match : matches)
        {
            cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
            cv::Point2f pt2 = keypoints2[match.trainIdx].pt + cv::Point2f(0, frame39.rows);

            cv::Mat ptHomogeneous = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);
            cv::Mat ptTransformed = bestHomography * ptHomogeneous;
            double w = ptTransformed.at<double>(2);
            cv::Point2f ptProjected(ptTransformed.at<double>(0) / w, ptTransformed.at<double>(1) / w + frame39.rows);

            if (cv::norm(ptProjected - pt2) < reprojectionThreshold)
            {
                cv::line(res, pt1, pt2, cv::Scalar(0, 255, 0)); // Green for inliers
            }
            else
            {
                cv::line(res, pt1, pt2, cv::Scalar(0, 0, 255)); // Red for outliers
            }
        }
        return res;
    }

    void FeatureExtractionAndMatching::show()
    {
        std::cout << "[INFO] Displaying images" << std::endl;
        cv::Mat part1 = drawPart1();
        cv::Mat part2 = drawPart2();
        cv::imshow("matches.jpg", part1);
        cv::imshow("inliers_outliers.jpg", part2);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    void FeatureExtractionAndMatching::run() // Run All
    {
        std::cout << "[INFO] Run Begin" << std::endl;
        SIFT();
        match();
        RANSAC();
        show();
    }
};
// CORE Part 1/2
int main()
{
    srand((unsigned)time(0)); // Seed for random number generation
    fem::FeatureExtractionAndMatching CORE;
    CORE.run();
    return 0;
}