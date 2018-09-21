#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <utils.hpp>

#define ROWS 480
#define COLS 640

using namespace std;
using namespace cv;

void get_background_models(std::vector<Mat>, std::vector<Mat>, Mat&, Mat&);

int main()
{
    int start_frame = 355;
    int end_frame = 420;

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

    Mat dummy_img(ROWS, COLS, CV_8UC1, Scalar(0));
    imshow("RGB", dummy_img);
    imshow("Depth", dummy_img);

    std::vector<Mat> depth_images;
    std::vector<Mat> rgb_images;

    Mat depth_model;
    Mat rgb_model;

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
        Mat color(Size(COLS, ROWS), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        if(source == 1)
            cv::cvtColor(color, color, cv::COLOR_BGR2RGB);

	    // Create depth image
        Mat depth16(Size(COLS, ROWS), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);

        if(frame_count >= start_frame && frame_count <= end_frame)
        {
            depth_images.push_back(depth16);
            rgb_images.push_back(color);
        }

        if(frame_count >= end_frame)
        {
            cout << "Computing background ...\n";
            get_background_models(depth_images, rgb_images, depth_model, rgb_model);

            // Depth to 8 bit for visualization
            Mat depth8u;
            depthMat16U_2_mat8U(depth_model, depth8u);
            imshow("Depth", depth8u);
            waitKey(0);
            break;
        }

        // Depth to 8 bit for visualization
        Mat depth8u;
        depthMat16U_2_mat8U(depth16, depth8u);

        // Display in a GUI
        imshow("RGB", color);
        imshow("Depth", depth8u);

        std::cout << "frame_count:" << frame_count++ << "\n";

        if(waitKey(5)>=0)
            break;
    }

    // Save models to disk
    cout << "Saving depth background model \n";
    serializeMatbin(depth_model, "depth_model.matbin");

    // Load model and show it
    Mat depth_model_saved = deserializeMatbin("depth_model.matbin");
    Mat depth8u;
    depthMat16U_2_mat8U(depth_model_saved, depth8u);
    imshow("Model saved", depth8u);
    waitKey(0);


    return 0;
}

void get_background_models(std::vector<Mat> depth_images,
                           std::vector<Mat> rgb_images,
                           Mat &depth_model, Mat &rgb_model)
{
    Mat img_i = depth_images[0];
    depth_model = cv::Mat(img_i.rows, img_i.cols, CV_32FC1);

    int num_images = depth_images.size();
    cout << "Num images:" << num_images << "\n";

    for(int img_idx=0; img_idx<num_images; img_idx++)
    {
        Mat img_i = depth_images[img_idx];

        int nl= img_i.rows; // number of lines
        int nc= img_i.cols ; // number of columns

        // if the input image is continuous
        // process it in a single larger loop for efficiency
        if(img_i.isContinuous())
        {
            nc = nc * nl;
            nl = 1;
        }

        // this loop is executed is executed only once for continuous input image
        for (int j=0; j<nl; j++) 
        {
            ushort* data16u= img_i.ptr<ushort>(j);
            float* dataSingle= depth_model.ptr<float>(j);

            for (int i=0; i<nc; i++) 
            {
                float bg = static_cast<float>(data16u[i]);
                dataSingle[i] += bg;
            } 
        }
    }

    depth_model/=num_images;
    depth_model.convertTo(depth_model, CV_16U);
}



