#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <utils.hpp>
#include <water_filling.hpp>

#define ROWS 480
#define COLS 640

using namespace std;
using namespace cv;


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

    Mat dummy_img(ROWS, COLS, CV_8UC1, Scalar(0));
    imshow("RGB", dummy_img);
    imshow("Depth", dummy_img);
    imshow("Seg", dummy_img);
    waitKey(0);

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
        Mat color(Size(COLS, ROWS), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        if(source == 1)
            cv::cvtColor(color, color, cv::COLOR_BGR2RGB);

	    // Create depth image
        Mat depth16(Size(COLS, ROWS), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);

        Mat wat_fill;
        wat_fill = waterfilling(depth16, 10);

        double min, max;
        cv::minMaxLoc(wat_fill, &min, &max);
        cout << "Min:" << min << '\n' << "Max:" << max << '\n';
        cv::inRange(wat_fill, 1000, 20000, wat_fill);

        // find contours
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(wat_fill, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        int nBlobs = 0;
        for (std::vector<std::vector<cv::Point> >::iterator c_iter = contours.begin(); c_iter != contours.end(); c_iter++) {
            double area = cv::contourArea(*c_iter);
            if (area > 300) {
                nBlobs++;
            } else {
                contours.erase(c_iter--);
            }
        }
        cv::Mat drawingMat = cv::Mat::zeros(ROWS, COLS, CV_8UC1);
        for (auto i = 0; i < contours.size(); ++i) {
            cv::drawContours(drawingMat, contours, i, 255, -1, 8);
        }
        std::cout << nBlobs << std::endl;

        // Depth to 8 bit for visualization
        Mat depth8u;
        depthMat16U_2_mat8U(depth16, depth8u);

        // Display in a GUI
        imshow("RGB", color);
        imshow("Depth", depth8u);
        imshow("Seg", wat_fill);
        imshow("Contours", drawingMat);

        if(waitKey(5)>=0)
            break;
    }
    return 0;
}



