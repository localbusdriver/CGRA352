#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>

const int IMAGES_PER_ROW = 17;
const float SCALE_U = 30.0, SCALE_V = 30.0;

std::vector<cv::Mat> loadImages(const char *path)
{
  std::cout << "Loading light field ..." << std::endl;
  std::vector<cv::String> filename;
  std::vector<cv::Mat> lf_imgs;
  cv::glob(path, filename);

  for (cv::String cv_str : filename)
  {
    std::string filepath(cv_str);
    size_t pos = filepath.find_last_of("/\\");
    if (pos == std::string::npos)
    {
      std::cerr << "Invalid file name format: " << filepath << std::endl;
      std::cerr << "Expected in the form : [prefix]/[name]_[row]_[col]_[v]_[u][suffix]";
      continue;
    }

    std::string filename = filepath.substr(pos + 1);
    pos = 0;
    while ((pos = filename.find("_", pos)) != std::string::npos)
    {
      filename.replace(pos, 1, " ");
      pos++;
    }

    std::istringstream ss(filename);
    std::string name;
    int row, col;
    float v, u;
    ss >> name >> row >> col >> v >> u;

    if (ss.good())
    {
      lf_imgs.push_back(cv::imread(filepath));
      continue;
    }
  }
  std::cout << "Finished loading light field" << std::endl;
  return lf_imgs;
}

// CORE Part 1
// Function to load light field images and access a specific pixel
cv::Vec3b accessLightFieldPixel(const std::vector<cv::Mat> &lightField, int row, int col, int t, int s)
{
  // Calculate the index of the image in the vector
  int index = row * 17 + col;
  cv::Mat img = lightField.at(index);        // Access the image
  cv::Vec3b pixel = img.at<cv::Vec3b>(t, s); // Access the pixel
  return pixel;
}

// CORE Part 2
// Check if a given UV coordinate is within the aperture radius with epsilon
bool isWithinAperture(float u, float v, float centerU, float centerV, float radius)
{
  const float EPSILON = 0.0001f;
  float dist = sqrt((u - centerU) * (u - centerU) + (v - centerV) * (v - centerV)); // Calculate distance
  std::cout << "Distance: " << dist << " | Radius+EPSILON: " << radius + EPSILON << std::endl;
  return dist <= radius + EPSILON; // Check if distance is less than radius + epsilon
}
// Function to construct UV-image for given ST coordinates with specific aperture
cv::Mat constructUVImage(const std::vector<cv::Mat> &lightField, int t, int s, float centerU, float centerV, float radius)
{
  cv::Mat uvImage(17, 17, CV_8UC3, cv::Scalar(0, 0, 0)); // Black image for non-valid pixels
  for (int row = 0; row < IMAGES_PER_ROW; ++row)
  {
    for (int col = 0; col < IMAGES_PER_ROW; ++col)
    {
      float u = centerU + (col - 8) * SCALE_U;
      float v = centerV + (row - 8) * SCALE_V;
      if (isWithinAperture(u, v, centerU, centerV, radius)) // Check dist from aperture center
      {
        uvImage.at<cv::Vec3b>(row, col) = lightField[row * 17 + col].at<cv::Vec3b>(t, s); // Copy pixel value
      }
    }
  }
  return uvImage;
}

void generateSTArray(std::vector<cv::Mat> lf_imgs)
{
  int startTime = time(NULL);
  int startS = 770, endS = 870;
  int startT = 205, endT = 305;
  int width = (endS - startS) * 17;
  int height = (endT - startT) * 17;
  cv::Mat largeImage(height, width, CV_8UC3);

  float centerU = -776.880371;
  float centerV = 533.057190;
  float radius = 75;

  for (int t = startT; t < endT; ++t)
  {
    for (int s = startS; s < endS; ++s)
    {
      cv::Mat uvImage = constructUVImage(lf_imgs, t, s, centerU, centerV, radius);
      uvImage.copyTo(largeImage(cv::Rect((s - startS) * 17, (t - startT) * 17, 17, 17)));
    }
  }
  int endTime = time(NULL);
  std::cout << "\nTime taken to construct UV image: " << endTime - startTime << " seconds" << std::endl;
  cv::imshow("Aperture Radius 75", largeImage);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

cv::Mat refocus(const std::vector<cv::Mat> &lightField, float focalDepth)
{
  int numImages = lightField.size();
  int width = lightField[0].cols;
  int height = lightField[0].rows;
  int numRows = 17, numCols = 17;
  cv::Mat outputImage = cv::Mat::zeros(height, width, CV_8UC3);

  for (int t = 0; t < height; ++t)
  {
    for (int s = 0; s < width; ++s)
    {
      cv::Vec3f accumulatedColor = cv::Vec3f(0, 0, 0);
      int count = 0;

      // Collect rays from different viewpoints
      for (int row = 0; row < numRows; ++row)
      {
        for (int col = 0; col < numCols; ++col)
        {
          int centerOffsetRow = row - (numRows / 2);
          int centerOffsetCol = col - (numCols / 2);
          int adjustedT = t + static_cast<int>(focalDepth * centerOffsetRow); // Convert floating point to integer
          int adjustedS = s + static_cast<int>(focalDepth * centerOffsetCol);

          // Check if adjusted coordinates are within bounds
          if (adjustedT >= 0 && adjustedT < height && adjustedS >= 0 && adjustedS < width)
          {
            cv::Vec3b color = accessLightFieldPixel(lightField, row, col, adjustedT, adjustedS);
            accumulatedColor += cv::Vec3f(color[0], color[1], color[2]);
            count++;
          }
        }
      }

      // Average the accumulated color from all valid rays and set it to the output image
      if (count > 0)
      {
        accumulatedColor /= count;
        unsigned char b = static_cast<unsigned char>(accumulatedColor[0]);
        unsigned char g = static_cast<unsigned char>(accumulatedColor[1]);
        unsigned char r = static_cast<unsigned char>(accumulatedColor[2]);
        outputImage.at<cv::Vec3b>(t, s) = cv::Vec3b(b, g, r);
      }
    }
  }
  return outputImage;
}

void incrementalFindBest(std::vector<cv::Mat> lf_imgs) // To find the best focal depth using incremental approach
{
  int start = time(NULL);
  std::vector<cv::Mat> lightField = lf_imgs;
  std::cout << "\nTrying different focal depths" << std::endl;

  cv::Mat focusedImage = refocus(lightField, 0);
  std::string filename = "focused_image_" + std::to_string(0) + ".png";
  cv::imwrite(filename, focusedImage);

  for (float i = -3.5; i > -4.5; i -= 0.1)
  {
    cv::Mat focusedImage = refocus(lightField, i);
    std::string filename = "focused_image_" + std::to_string(i) + ".png";
    cv::imwrite(filename, focusedImage);

    std::cout << "Focal depth: " << i << std::endl;
    std::cout << "Focal stack images saved as " << filename << " in root of exe\n"
              << std::endl;
  }

  int end = time(NULL);
  std::cout << "Time taken to run focal stack: " << end - start << " seconds" << std::endl;
}

void runFocalStack(std::vector<cv::Mat> lf_imgs)
{
  int start = time(NULL);
  std::vector<cv::Mat> lightField = lf_imgs;
  std::vector<float> focalDepths = {0, 1, 2, 3, 4};
  std::cout << "\nTrying different focal depths" << std::endl;
  for (size_t i = 0; i < focalDepths.size(); ++i)
  {
    cv::Mat focusedImage = refocus(lightField, focalDepths[i]);
    std::string filename = "focused_image_" + std::to_string(i + 1) + ".png";
    cv::imwrite(filename, focusedImage);

    std::cout << "Focal depth: " << focalDepths[i] << std::endl;
    std::cout << "Focal stack images saved as " << filename << " in root of exe\n"
              << std::endl;
  }
  int end = time(NULL);
  std::cout << "Time taken to run focal stack: " << end - start << " seconds" << std::endl;
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " <path to light field images folder>\n";
    return EXIT_FAILURE;
  }

  std::cout << "Light field image path: " << argv[1] << std::endl;
  std::vector<cv::Mat> lf_imgs = loadImages(argv[1]);

  // CORE part 1
  int row = 7, col = 10, t = 384, s = 768;
  cv::Vec3b pixel = accessLightFieldPixel(lf_imgs, row, col, t, s);
  std::cout << "Pixel value at (7, 10, 384, 768): ["
            << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << "]" << std::endl;

  // CORE part 2
  generateSTArray(lf_imgs);

  // COMPLETION / CHALLENGE
  // incrementalFindBest(lf_imgs);
  runFocalStack(lf_imgs);

  return 0;
}