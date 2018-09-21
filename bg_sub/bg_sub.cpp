#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <utils.hpp>

#define ROWS 480
#define COLS 640

using namespace std;
using namespace cv;


int main()
{
    int source = 1;  // 0 for video stream, 1 for bag
    string bag_file = "/datasets/Real_sense/14_08_18/vid1.bag";

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

    // Setup the aligned mechanism
    rs2::align align(RS2_STREAM_COLOR);


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
    namedWindow("Model", WINDOW_AUTOSIZE);
    namedWindow("Forg", WINDOW_AUTOSIZE);
    namedWindow("ForgFinal", WINDOW_AUTOSIZE);
    namedWindow("Mask", WINDOW_AUTOSIZE);

    Mat dummy_img(ROWS, COLS, CV_8UC1, Scalar(0));
    imshow("RGB", dummy_img);
    imshow("Depth", dummy_img);

    // Load background model and show it
    Mat depth_model = deserializeMatbin("depth_model_aligned.matbin");
    Mat depth8u;
    depthMat16U_2_mat8U(depth_model, depth8u);
    imshow("Model", depth8u);
    waitKey(0);

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
        //rs2::frame color_frame = frames.get_color_frame();
        //rs2::frame depth_frame = frames.get_depth_frame();

        // Get aligned color and depth
        auto aligned_frames = align.process(frames);
        rs2::video_frame color_frame = aligned_frames.first(RS2_STREAM_COLOR);
        rs2::depth_frame depth_frame = aligned_frames.get_depth_frame();


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

        // Depth to 8 bit for visualization
        Mat depth8u, forg8u;
        depthMat16U_2_mat8U(depth16, depth8u);
        depthMat16U_2_mat8U(forg, forg8u);

        cv::Mat forgFinal = cv::Mat::zeros(ROWS, COLS, CV_8UC1);
        depth8u.copyTo(forgFinal, mask);

        // Display in a GUI
        imshow("RGB", color);
        imshow("Depth", depth8u);
        imshow("Forg", forg8u);
        imshow("Mask", mask);
        imshow("ForgFinal", forgFinal);

        if(waitKey(0) == 27)
            break;
    }

    return 0;
}



