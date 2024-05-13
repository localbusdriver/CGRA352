#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <sstream>

// Function to load light field images and access a specific pixel
cv::Vec3b accessLightFieldPixel(const std::vector<cv::Mat> &lightField, int row, int col, int t, int s)
{
  // Calculate the index of the image in the vector
  int index = row * 17 + col;
  cv::Mat img = lightField.at(index);
  cv::Vec3b pixel = img.at<cv::Vec3b>(t, s);
  return pixel;
}

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
    if (pos != std::string::npos)
    {
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
        cv::imshow("Image", cv::imread(filepath));

        lf_imgs.push_back(cv::imread(filepath));
        continue;
      }
    }
    std::cerr << "Filepath error with : " << filepath << std::endl;
    std::cerr << "Expected in the form : [prefix]/[name]_[row]_[col]_[v]_[u][suffix]";
    abort();
  }
  std::cout << "Finished loading light field" << std::endl;
  return lf_imgs;
}

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <path to light field images folder>" << std::endl;
    return 1;
  }
  std::cout << "Light field image path: " << argv[1] << std::endl;
  std::vector<cv::Mat> lf_imgs = loadImages(argv[1]);

  int row = 7, col = 10, t = 384, s = 768;
  cv::Vec3b pixel = accessLightFieldPixel(lf_imgs, row, col, t, s);
  std::cout << "Pixel value at (7, 10, 384, 768): ["
            << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << "]" << std::endl;

  return 0;
}


