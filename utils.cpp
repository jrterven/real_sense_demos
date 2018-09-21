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
			uchar intensity = static_cast<uchar>(data16u[i] >> 5);
			data8u[i] = intensity;

		} // end of line
	}
}

/* Converts a 16U color-aligned depth image to an 8UC3 with color on the valid depth pixels
*/
void depth16U_2_depthColored8UC3(cv::Mat cvDepthFrame16u, cv::Mat cvColorFrame, cv::Mat &cvDepthColored)
{
	int nl = cvDepthFrame16u.rows; // number of lines
	int nc = cvDepthFrame16u.cols ; // number of columns

	cvDepthColored = cv::Mat::zeros(nl,nc,CV_8UC3);

	// if the input image is continuous
	// process it in a single larger loop for efficiency
	if(cvDepthFrame16u.isContinuous() && cvColorFrame.isContinuous())
	{
		nc = nc * nl;
		nl = 1;
	}

    nc = nc*3;

	// this loop is executed is executed only once for continuous input image
	for (int j=0; j<nl; j++) 
	{
		// get the address of row j
		uchar* dataColor= cvColorFrame.ptr<uchar>(j);
        uchar* dataColoredDepth= cvDepthColored.ptr<uchar>(j);
		ushort* data16u= cvDepthFrame16u.ptr<ushort>(j);

		for (int i=0, i3=0 ; i3<nc; i++, i3+=3)
		{
			if(data16u[i] != 0)
            {
                dataColoredDepth[i3] = dataColor[i3];
                dataColoredDepth[i3+1] = dataColor[i3+1];
                dataColoredDepth[i3+2] = dataColor[i3+2];
            }
		} // end of line
	}
}

void serializeMatbin(cv::Mat& mat, std::string filename)
{
    int elemSizeInBytes = (int)mat.elemSize();
    int elemType        = (int)mat.type();
    int dataSize        = (int)(mat.cols * mat.rows * mat.elemSize());

    FILE* FP = fopen(filename.c_str(), "wb");
    int sizeImg[4] = {mat.cols, mat.rows, elemSizeInBytes, elemType };
    fwrite(/* buffer */ sizeImg, /* how many elements */ 4, /* size of each element */ sizeof(int), /* file */ FP);
    fwrite(mat.data, mat.cols * mat.rows, elemSizeInBytes, FP);
    fclose(FP);
}

cv::Mat deserializeMatbin(std::string filename)
{
    FILE* fp = fopen(filename.c_str(), "rb");
    int header[4];
    fread(header, sizeof(int), 4, fp);
    int cols            = header[0]; 
    int rows            = header[1];
    int elemSizeInBytes = header[2];
    int elemType        = header[3];

    std::cout << "rows="<<rows<<" cols="<<cols<<" elemSizeInBytes="
			  << elemSizeInBytes << " elemType=" << elemType << std::endl;

    cv::Mat outputMat = cv::Mat::ones(rows, cols, elemType);

    size_t result = fread(outputMat.data, elemSizeInBytes, (size_t)(cols * rows), fp);

    if (result != (size_t)(cols * rows)) {
        fputs ("Reading error", stderr);
    }

    fclose(fp);
    return outputMat;
}