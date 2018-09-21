#include <opencv2/opencv.hpp>

void depthMat16U_2_mat8U(cv::Mat cvDepthFrame16u, cv::Mat &cvDepthFrame8u);
void depth16U_2_depthColored8UC3(cv::Mat cvDepthFrame16u, cv::Mat cvColorFrame, cv::Mat &cvDepthColored);
void serializeMatbin(cv::Mat& mat, std::string filename);
cv::Mat deserializeMatbin(std::string filename);