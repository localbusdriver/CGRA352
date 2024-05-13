#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

vector<Mat> loadImage(string path)
{
  std::cout << "Loading light field ..." << std::endl;
  std::vector<cv::String> lf_imgs;
  // cv::glob(argv[1], lf_imgs);
  for (cv::String cv_str : lf_imgs)
  {
    // get the filepath
    std::string filepath(cv_str);
    size_t pos = filepath.find_last_of("/\\");
    if (pos != std::string::npos)
    {
      // replace "_" with " "
      std::string filename = filepath.substr(pos + 1);
      pos = 0;
      while ((pos = filename.find("_", pos)) != std::string::npos)
      {
        filename.replace(pos, 1, " ");
        pos++;
      }
      // parse for values
      std::istringstream ss(filename);
      std::string name;
      int row, col;
      float v, u;
      ss >> name >> row >> col >> v >> u;
      if (ss.good())
      {
        // TODO something with the image file "filepath"
        // TODO something with the coordinates: row, col, v, u
        continue;
      }
    }
    // throw error otherwise
    std::cerr << "Filepath error with : " << filepath << std::endl;
    std::cerr << "Expected in the form : [prefix]/[name]_[row]_[col]_[v]_[u][suffix]";
    abort();
  }
  std::cout << "Finished loading light field" << std::endl;
}

int main(int argc, char **argv)
{
  Mat image = imread("./rectified/out_00_00_-702.718018_459.879150.png", IMREAD_COLOR);
  cv::imshow("Image", image);
  cv::waitKey(0);
  printf("Hello World");
  return 0;
}