/*
    This script plays a bag file or a camera stream.
    Optionally, it can also save the frames in output_dir
*/
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <utils.hpp>

#define ROWS 480
#define COLS 640

using namespace std;
using namespace cv;


int main()
{
    int source = 0;  // 0 for video stream, 1 for bag
    bool save_frames = true;
    string output_dir = "/datasets/Real_sense/21_09_18";
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
        cfg.enable_device_from_file(bag_file, !save_frames);
    }

    //Instruct pipeline to start streaming with the requested configuration
    pipe.start(cfg);
    device = pipe.get_active_profile().get_device();
    rs2::frameset frames;

    // Setup the aligned mechanism
    rs2::align align(RS2_STREAM_COLOR);

    // Get depth camera intrinsic parameters
    auto const intrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
    std::cout << "fx, fy = " << intrinsics.fx << "," << intrinsics.fy << "\n";
    std::cout << "cx, cy = " << intrinsics.ppx << "," << intrinsics.ppy << "\n";
    std::cout << "dist:" << intrinsics.coeffs << "\n";


    if(source == 0)
    {
        // Camera warmup - dropping several first frames to let auto-exposure stabilize
        
        for(int i = 0; i < 80; i++)
        {
            //Wait for all configured streams to produce a frame
            frames = pipe.wait_for_frames();
        }
    }

    namedWindow("RGB", WINDOW_AUTOSIZE );
    namedWindow("Depth", WINDOW_AUTOSIZE );
    namedWindow("Depth Aligned", WINDOW_AUTOSIZE );
    namedWindow("Colored depth", WINDOW_AUTOSIZE);

    Mat dummy_img(ROWS, COLS, CV_8UC1, Scalar(0));
    imshow("RGB", dummy_img);
    imshow("Depth", dummy_img);

    // Compression params for imwrite
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);


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

        //Get not aligned depth frame
        rs2::frame depth_frame = frames.get_depth_frame();

        // Get aligned color and depth
        auto aligned_frames = align.process(frames);
        rs2::video_frame color_frame = aligned_frames.first(RS2_STREAM_COLOR);
        rs2::depth_frame aligned_depth_frame = aligned_frames.get_depth_frame();

        // Create OpenCV color image
        Mat color(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        if(source == 1)
            cv::cvtColor(color, color, cv::COLOR_BGR2RGB);

	    // Create depth image
        Mat depth16(Size(640, 480), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        Mat depth16alg(Size(640, 480), CV_16U, (void*)aligned_depth_frame.get_data(), Mat::AUTO_STEP);

        // Depth to 8 bit for visualization
        Mat depth8u, depth8ualg;
        depthMat16U_2_mat8U(depth16, depth8u);
        depthMat16U_2_mat8U(depth16alg, depth8ualg);

        // Colored depth
        Mat coloredDepth;
        depth16U_2_depthColored8UC3(depth16alg, color, coloredDepth);

        // Display in a GUI
        imshow("RGB", color);
        imshow("Depth", depth8u);
        imshow("Depth Aligned", depth8ualg);
        imshow("Colored depth", coloredDepth);

        //std::cout << "frame_count:" << frame_count++ << "\n";

        if(save_frames)
        {
            imwrite(output_dir + "/" + to_string(frame_count) + "c.png", color);
            imwrite(output_dir + "/" + to_string(frame_count) + "cd.png", coloredDepth, compression_params);
            imwrite(output_dir + "/" + to_string(frame_count) + "d8.png", depth8ualg, compression_params);
            serializeMatbin(depth16alg, output_dir + "/" + to_string(frame_count) + "d16.matbin");
        }

        frame_count++;
        if(waitKey(5)>=0)
            break;
    }

    return 0;
}





