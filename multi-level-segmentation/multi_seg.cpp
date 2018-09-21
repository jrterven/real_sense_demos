#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <utils.hpp>

#define ROWS 480
#define COLS 640

using namespace std;
using namespace cv;

void mls(Mat &depthImage, Mat &output);

int main()
{
    int source = 1;  // 0 for video stream, 1 for bag
    string bag_file = "/datasets/Real_sense/14_08_18/vid2.bag";

    //Contruct a pipeline which abstracts the device
    rs2::pipeline pipe;

    // Initialize a shared pointer to a device with the current device on the pipeline
    rs2::device device;

    //Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    if(source == 0)
    {
        //Add desired streams to configuration
        cfg.enable_stream(RS2_STREAM_COLOR, COLS, ROWS, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, COLS, ROWS, RS2_FORMAT_Z16, 30);
    }
    else
    {
        cfg.enable_device_from_file(bag_file);
    }

    //Instruct pipeline to start streaming with the requested configuration
    pipe.start(cfg);

    device = pipe.get_active_profile().get_device();

    rs2::frameset frames;

    if(source == 0)
    {
        // Camera warmup - dropping several first frames to let auto-exposure stabilize
        
        for(int i = 0; i < 30; i++)
        {
            //Wait for all configured streams to produce a frame
            frames = pipe.wait_for_frames();
        }
    }

    namedWindow("RGB", WINDOW_AUTOSIZE );
    namedWindow("Depth", WINDOW_AUTOSIZE );
    namedWindow("ForgFinal", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);

    Mat dummy_img(ROWS, COLS, CV_8UC1, Scalar(0));
    imshow("RGB", dummy_img);
    imshow("Depth", dummy_img);
    imshow("ForgFinal", dummy_img);
    imshow("Output", dummy_img);

    // Load background model and show it
    Mat depth_model = deserializeMatbin("depth_model.matbin");
    Mat depth8u;
    depthMat16U_2_mat8U(depth_model, depth8u);

    // Create a structuring element
    int erosionSize = 3;
    // Selecting a elliptical kernel 
    Mat element = getStructuringElement(MORPH_ELLIPSE, 
                                        Size(2 * erosionSize + 1, 2 * erosionSize + 1),
                                        Point(erosionSize, erosionSize));

    int frame_count = 0;
    while(1)
    {
        // check if streamed frame is ready
        if(source == 0)
            frames = pipe.wait_for_frames();
        // check if recorded frame is ready
        else if (source == 1 && !pipe.poll_for_frames(&frames))
        {
            waitKey(5);
            continue;
        }
            
        //Get each frame
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();

        // Create OpenCV color image
        Mat color(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        if(source == 1)
            cv::cvtColor(color, color, cv::COLOR_BGR2RGB);

	    // Create depth image
        Mat depth16(Size(640, 480), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);

        // Subtract the background model
        Mat forg = depth_model - depth16;
        Mat forgbin;
        threshold(forg, forgbin, 400, 255, THRESH_BINARY);
        forgbin.convertTo(forgbin, CV_8U);
        std::vector<std::vector<cv::Point> > contours;


        cv::findContours(forgbin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        for (int i = 0; i < contours.size(); ++i)
        {
            int area = cv::contourArea(contours[i]);
            
            if (area < 2000.0)
            {
                contours.erase(contours.begin() + i--);
            }
        }
        cout << contours.size() << "\n";
        cv::Mat mask = cv::Mat::zeros(ROWS, COLS, CV_8UC1);
        cv::drawContours(mask, contours, -1, 255, -1);
        morphologyEx(mask, mask, MORPH_OPEN, element, Point(-1,-1), 2); 

        cv::Mat forgFinal = cv::Mat::zeros(ROWS, COLS, CV_8UC1);
        depth16.copyTo(forgFinal, mask);

        Mat output;
        mls(forgFinal, output);

        // Depth to 8 bit for visualization
        Mat forg8u, output8b;
        depthMat16U_2_mat8U(depth16, depth8u);
        depthMat16U_2_mat8U(forgFinal, forg8u);
        output.convertTo(output8b, CV_8U);
        addWeighted(forg8u, 0.8, output8b, 0.5, 0, output8b);


        // Display in a GUI
        imshow("RGB", color);
        imshow("Depth", depth8u);
        imshow("ForgFinal", forg8u);
        imshow("Output", output8b);

        if(waitKey(5)>=0)
            break;
    }

    return 0;
}

// Multi-level segmentation
void mls(Mat &depthImage, Mat &drawingMat)
{
    std::vector<cv::Point> pheadsVec;
    std::vector<std::vector<cv::Point> > contours;

    
    for (auto i=0; i<depthImage.rows; ++i) 
    {
        for (auto j=0; j<depthImage.cols; ++j) 
        {
            if (depthImage.at<unsigned short>(cv::Point(j,i))) 
            {
                depthImage.at<unsigned short>(cv::Point(j,i)) =
                        (depthImage.at<unsigned short>(cv::Point(j,i)) -4000)*-1;
            }
        }
    }
    
    // find global max point
    double maxHeight;
    cv::Point p_maxHeight;
    cv::minMaxLoc(depthImage, NULL, &maxHeight, NULL, &p_maxHeight);
    //cout << "MaxHeight:" << maxHeight << " Location:" << p_maxHeight.x << "," << p_maxHeight.y << "\n";

    // begin multi-level segmentation
    std::vector<std::vector<cv::Point> > contoursVector;
    int thresholdLevel = 40;
    int minHeight = 300;

    for (auto level = maxHeight - thresholdLevel;
         level > minHeight;
         level -= thresholdLevel) {
        // find all points over the level
        cv::Mat thresholdLevelMat = depthImage > level;
        std::vector<std::vector<cv::Point> > contoursTemp;
        /// find contours of this level
        cv::findContours(thresholdLevelMat,
                         contoursTemp,
                         CV_RETR_EXTERNAL,
                         CV_CHAIN_APPROX_SIMPLE);
        // filter small contours and not round contours                         
        for (auto i = 0; i < contoursTemp.size(); ++i) {
            auto area = cv::contourArea(contoursTemp[i]);
            //auto perimeter = cv::arcLength(contoursTemp[i], true);
            //auto T = 4 * 3.14159265 * (area/(perimeter*perimeter));
            //if (area < 500 || T < 0.7)
            if (area < 500)
                contoursTemp.erase(contoursTemp.begin() + i--);
        }
        // filter not round contours


        // inserts in the vector points the highest point/depth value 
        // of each blob identified by means of the FilterMask function
        for (auto k = 0; k < contoursTemp.size(); ++k) {
            cv::Mat maskcont = cv::Mat::zeros(depthImage.rows,
                                              depthImage.cols,
                                              CV_8UC1);
            cv::Mat depthMatFilterCont = cv::Mat::zeros(depthImage.rows,
                                                        depthImage.cols,
                                                        CV_16UC1);
            cv::drawContours(maskcont, contoursTemp, k, 255, -1);
            depthImage.copyTo(depthMatFilterCont, maskcont);

            /// find max Point
            double maxblobValue;
            cv::Point maxblobPoint;
            cv::minMaxLoc(depthMatFilterCont,
                          NULL,
                          &maxblobValue,
                          NULL,
                          &maxblobPoint);

            auto pointCheck = false;
            for (auto m = 0; m < pheadsVec.size(); ++m) {
                if (maxblobPoint == pheadsVec[m]) {
                    pointCheck = true;
                }
            }
            if (!pointCheck) {
                pheadsVec.push_back(maxblobPoint);
                contoursVector.push_back(contoursTemp[k]);
            }
        }
    }
    std::cout << contoursVector.size() << std::endl;
    drawingMat = cv::Mat::zeros(depthImage.rows, depthImage.cols, CV_8UC1);
    for (auto i = 0; i < contoursVector.size(); ++i) {
        cv::drawContours(drawingMat, contoursVector, i, 255, -1, 8);
    }
}
