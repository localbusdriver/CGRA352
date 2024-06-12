#include "main.hpp"
#include <filesystem>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;
namespace cw
{
    void CroppingWindow::transformMask()
    {
        std::cout << "[INFO] Transforming masks..." << std::endl;
        int width = frames[0].cols;
        int height = frames[0].rows;
        cv::Mat whiteMask = cv::Mat::ones(height, width, CV_8UC1) * 255; // Initialize mask with all white pixels
        std::vector<cv::Mat> transformedMasks;
        for (const auto &H : homographies)
        {
            cv::Mat transformedMask;
            cv::warpPerspective(whiteMask, transformedMask, H, cv::Size(width, height)); // Apply homography to mask
            masks.push_back(transformedMask);
        }
    };

    cv::Mat CroppingWindow::generateCombinedMask()
    {
        std::cout << "[INFO] Generating combined mask..." << std::endl;
        cv::Mat combinedMask = masks[0].clone();

        for (const auto &mask : masks)
            cv::bitwise_and(combinedMask, mask, combinedMask); // Keep only valid pixels
        return combinedMask;
    };

    cv::Rect CroppingWindow::findLargestInscribedSquare(const cv::Mat &mask)
    {
        std::cout << "[INFO] Finding largest inscribed square..." << std::endl;

        int rows = mask.rows;
        int cols = mask.cols;
        cv::Mat S = cv::Mat::zeros(rows, cols, CV_32SC1); // Matrix to store the size of largest square at each pixel

        int maxSize = 0;
        cv::Point maxPoint;
        // Iterate over the mask from bottom-right to top-left
        for (int y = rows - 1; y >= 0; --y)
        {
            for (int x = cols - 1; x >= 0; --x)
            {
                if (mask.at<uchar>(y, x) == 255) // White Pixel
                {
                    if (y == rows - 1 || x == cols - 1)
                    {
                        // If the pixel is on the bottom or right edge, it can only form a 1x1 square
                        S.at<int>(y, x) = 1; // Last row or column
                    }
                    else
                    {
                        // Calculate the size of the largest square ending at this pixel
                        S.at<int>(y, x) = std::min({S.at<int>(y + 1, x), S.at<int>(y, x + 1), S.at<int>(y + 1, x + 1)}) + 1;
                    }
                    if (S.at<int>(y, x) > maxSize)
                    {
                        maxSize = S.at<int>(y, x);
                        maxPoint = cv::Point(x, y);
                    }
                }
            }
        }
        std::cout << "[INFO] Largest inscribed square: " << maxSize << "x" << maxSize << " (800x450) at (" << maxPoint.x << ", " << maxPoint.y << ")" << std::endl;
        cv::Rect largestSquare = cv::Rect(maxPoint.x, maxPoint.y, maxSize, maxSize);
        return largestSquare;
    };

    cv::Rect CroppingWindow::findLargestInscribedRectangle(const cv::Mat &mask)
    {
        std::cout << "[INFO] Finding largest inscribed rectangle..." << std::endl;

        int rows = mask.rows;
        int cols = mask.cols;

        cv::Mat width = cv::Mat::zeros(rows, cols, CV_32SC1);
        cv::Mat height = cv::Mat::zeros(rows, cols, CV_32SC1);

        int maxArea = 0;
        cv::Rect maxRect;

        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                if (mask.at<uchar>(y, x) == 255)
                {
                    if (x == 0)
                    {
                        width.at<int>(y, x) = 1;
                    }
                    else
                    {
                        width.at<int>(y, x) = width.at<int>(y, x - 1) + 1;
                    }

                    if (y == 0)
                    {
                        height.at<int>(y, x) = 1;
                    }
                    else
                    {
                        height.at<int>(y, x) = height.at<int>(y - 1, x) + 1;
                    }

                    int minWidth = width.at<int>(y, x);

                    // Find the largest rectangle with the current pixel as the top-right corner
                    for (int h = 0; h < height.at<int>(y, x); ++h)
                    {
                        // minimum width of the rectangle
                        minWidth = std::min(minWidth, width.at<int>(y - h, x));
                        int area = (h + 1) * minWidth;
                        if (area > maxArea)
                        {
                            maxArea = area;
                            maxRect = cv::Rect(x - minWidth + 1, y - h, minWidth, h + 1);
                        }
                    }
                }
            }
        }

        std::cout << "[INFO] Largest inscribed rectangle: " << maxRect.width << "x" << maxRect.height << " at (" << maxRect.x << ", " << maxRect.y << ")" << std::endl;

        return maxRect;
    }

    void CroppingWindow::run()
    {
        createDir();
        transformMask();
        cv::Mat combinedMask = generateCombinedMask();
        // cv::Rect largestSquare = findLargestInscribedSquare(combinedMask);
        cv::Rect largestRect = findLargestInscribedRectangle(combinedMask);
        // cropAndSaveFrames(largestSquare);
        cropAndSaveFrames(largestRect);
        show();
    };

    void CroppingWindow::cropAndSaveFrames(const cv::Rect &cropRect)
    {
        std::cout << "[INFO] Cropping and saving frames..." << std::endl;

        /** CROP and SAVE */
        for (size_t i = 0; i < frames.size(); i++)
        {
            cv::Mat croppedFrame = frames[i](cropRect);
            std::string num = std::to_string(i);
            num = std::string(3 - num.length(), '0') + num; // e.g. if index is 3; num = "003"
            std::string filename = "./CHALLENGE/crop" + num + ".png";
            if (!cv::imwrite(filename, croppedFrame))
            {
                std::cerr << "[Error] Failed to save cropped frame " << i << std::endl;
            }
        }
    };

    void CroppingWindow::show()
    {
        std::cout << "[INFO] Displaying cropped frames..." << std::endl;
        for (size_t i = 0; i < frames.size(); i++)
        {
            std::string num = std::to_string(i);
            num = std::string(3 - num.length(), '0') + num; // e.g. if index is 3; num = "003"
            
            cv::Mat croppedImg = cv::imread("./CHALLENGE/crop" + num + ".png");
            cv::imshow("Cropped Frame", croppedImg);
            cv::waitKey(100);
        }
    }

    CroppingWindow::CroppingWindow()
    {
        std::cout << "\n[INFO] Initializing Cropping Window..." << std::endl;
        loadImages();
    };

    void CroppingWindow::createDir()
    {
        std::cout << "[INFO] Creating directory 'CHALLENGE' " << std::endl;
        fs::path dir = "./CHALLENGE";
        try
        { // Create directory
            if (fs::create_directory(dir))
                std::cout << "[INFO] Directory Created: " << dir << std::endl;
            else
                std::cout << "[INFO] Directory Already Exists: " << dir << std::endl;
        }
        catch (const fs::filesystem_error &e)
        {
            std::cerr << "Error creating directory: " << e.what() << '\n';
        }
    };

    void CroppingWindow::loadImages()
    {
        std::cout << "[INFO] Loading Images..." << std::endl;
        for (int i = 0; i < 101; ++i)
        {
            std::string num = std::to_string(i);
            num = std::string(3 - num.length(), '0') + num; // e.g. if index is 3; num = "003"

            std::string filename = "./COMPLETION/stable" + num + ".png";
            cv::Mat img = cv::imread(filename);

            if (img.empty())
            {
                std::cerr << "[Error] Stabilized Frame not found" << std::endl;
                exit(1);
            }
            frames.push_back(img);
        }

        for (int i = 0; i < 101; ++i)
        {
            std::string num = std::to_string(i);
            num = std::string(3 - num.length(), '0') + num; // e.g. if index is 3; num = "003"

            std::string filename = "./HOMOGRAPHIES/homography" + num + ".xml";
            cv::FileStorage fs(filename, cv::FileStorage::READ);
            cv::Mat H;
            fs["homography"] >> H;
            if (H.empty())
            {
                std::cerr << "[Error] Transformation matrix not found" << std::endl;
                exit(1);
            }
            homographies.push_back(H);
        }
    }

    // void CroppingWindow::loadImages()
    // {
    //     std::cout << "[INFO] Loading frames and homographies..." << std::endl;
    //     for (const auto &file : fs::directory_iterator("./COMPLETION"))
    //     {
    //         if (file.is_regular_file())
    //         {
    //             std::string extension = file.path().extension().string();
    //             if (extension == ".png" || extension == ".jpg")
    //             {
    //                 cv::Mat frame = cv::imread(file.path().string());
    //                 if (frame.empty())
    //                 {
    //                     std::cerr << "[Error] Image not found" << std::endl;
    //                     exit(1);
    //                 }
    //                 frames.push_back(frame);
    //             }
    //             if (extension == ".xml")
    //             {
    //                 cv::FileStorage fs(file.path().string(), cv::FileStorage::READ);
    //                 cv::Mat H;
    //                 fs["homography"] >> H;
    //                 if (H.empty())
    //                 {
    //                     std::cerr << "[Error] Transformation matrix not found" << std::endl;
    //                     exit(1);
    //                 }
    //                 homographies.push_back(H);
    //             }
    //         }
    //     }
    //     if (frames.size() != homographies.size())
    //     {
    //         std::cerr << "[Error] Number of frames(" << frames.size() << ") and homographies(" << homographies.size() << ") do not match" << std::endl;
    //         exit(1);
    //     }
    // };
}