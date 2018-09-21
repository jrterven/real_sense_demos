#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <utils.hpp>

using namespace std;
using namespace cv;


int main()
{
    //Contruct a pipeline which abstracts the device
    rs2::pipeline pipe;

    //Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    //Add desired streams to configuration
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    //Instruct pipeline to start streaming with the requested configuration
    pipe.start(cfg);

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    rs2::frameset frames;
    for(int i = 0; i < 30; i++)
    {
        //Wait for all configured streams to produce a frame
        frames = pipe.wait_for_frames();
    }

    namedWindow("RGB", WINDOW_AUTOSIZE );
    namedWindow("IR", WINDOW_AUTOSIZE );
    namedWindow("DEPTH", WINDOW_AUTOSIZE );

    while(1)
    {
        frames = pipe.wait_for_frames();

        //Get each frame
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame ir_frame = frames.first(RS2_STREAM_INFRARED);
        rs2::frame depth_frame = frames.get_depth_frame();

        // Creating OpenCV Matrix from a color image
        Mat color(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);

        // Creating OpenCV matrix from IR image
        Mat ir(Size(640, 480), CV_8UC1, (void*)ir_frame.get_data(), Mat::AUTO_STEP);

	    // Create depth image
        Mat depth16(Size(640, 480), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);

        // Apply Histogram Equalization to IR
        //equalizeHist( ir, ir );
        //applyColorMap(ir, ir, COLORMAP_JET);

        // Depth to 8 bit
        Mat depth8u;
        depthMat16U_2_mat8U(depth16, depth8u);

        // Display in a GUI
    
        imshow("RGB", color);
        imshow("IR", ir);
        imshow("DEPTH", depth8u);

        if(waitKey(5)>=0)
            break;
    }

    return 0;
}



