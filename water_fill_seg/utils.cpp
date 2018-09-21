#include <opencv2/opencv.hpp>
#include "utils.hpp"

/* Converts a 16U depth image to an 8UC1 (8bit, 1 channel) image suitable for visualization as grayscale image. 
    The conversion only returns from the 4th bit to the 12bith. These values where selected from empirical observations.
*/
void depthMat16U_2_mat8U(cv::Mat cvDepthFrame16u, cv::Mat &cvDepthFrame8u)
{
	int nl= cvDepthFrame16u.rows; // number of lines
	int nc= cvDepthFrame16u.cols ; // number of columns

	cvDepthFrame8u = cv::Mat(nl,nc,CV_8UC1);

	// if the input image is continuous
	// process it in a single larger loop for efficiency
	if(cvDepthFrame16u.isContinuous())
	{
		nc = nc * nl;
		nl = 1;
	}

	// this loop is executed is executed only once for continuous input image
	for (int j=0; j<nl; j++) 
	{
		// get the address of row j
		uchar* data8u= cvDepthFrame8u.ptr<uchar>(j);
		ushort* data16u= cvDepthFrame16u.ptr<ushort>(j);

		for (int i=0; i<nc; i++) 
		{
			uchar intensity = static_cast<uchar>(data16u[i] >> 4);
			data8u[i] = intensity;

		} // end of line
	}
}