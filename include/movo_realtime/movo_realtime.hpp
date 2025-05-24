#ifndef MOVO_REALTIME_HPP
#define MOVO_REALTIME_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include <humanoid_interfaces/msg/imu_msg.hpp>
#include <humanoid_interfaces/msg/movo_msg.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <utility>
#include <limits>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

class MovoRealtime : public rclcpp::Node
{

public:
    MovoRealtime();
    ~MovoRealtime();

    double scale = 1.00;

    vector<uchar> status;

    Mat prevImage;
    Mat currImage;

    Mat img_1, img_2;

    Mat currImage_c;

    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    vector<DMatch> good_matches;
    vector<Point2f> points1, points2;

    Mat E, R, t, mask;

    Mat R_f, t_f;

    Mat traj = Mat::zeros(600, 600, CV_8UC3);

    cv::Mat img;
    cv::Mat info;

    Mat K_M = Mat(3, 3, CV_64FC1);
    Mat D_M = Mat(1, 5, CV_64FC1);
    Mat R_M = Mat(3, 3, CV_64FC1);
    Mat P_M = Mat(3, 4, CV_64FC1);

    Mat NEW_K_M = Mat(3, 3, CV_64FC1);

    cv::Point2d new_focalLen;
    cv::Point2d new_prncPt;
    cv::Point2d focalLen;
    cv::Point2d prncPt;

    vector<Point2f> prevFeatures;
    vector<Point2f> currFeatures;

    void image_processing(const cv::Mat &img);

    void calibration_info()
    {
        String temp;
        double dtemp;
        double mtemp;
        double asc46;
        double K[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        double D[5] = {0, 0, 0, 0, 0};
        double R[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        double P[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int cnt;
        ifstream Calibration_Info("/home/kgh/robot_ws/src/ocam_ros2/config/camera.yaml");
        if (Calibration_Info.is_open())
        {
            for (int i = 0; i < 12; i++)
            {
                Calibration_Info >> temp;
            }
            for (int i = 0; i < 9; i++)
            {
                char TEXT[256];
                dtemp = 0;
                asc46 = 1;
                mtemp = 1;
                Calibration_Info >> temp;
                strcpy(TEXT, temp.c_str());
                for (std::string::size_type j = 0; j < temp.length(); j++)
                {
                    if (TEXT[j] == 45)
                    {
                        mtemp = -1;
                    }
                    if ((TEXT[j] >= 48 && TEXT[j] <= 57) || TEXT[j] == 46)
                    {
                        if (TEXT[j] == 46 || asc46 != 1)
                        {
                            if (TEXT[j] != 46)
                            {
                                dtemp += (TEXT[j] - 48) / asc46;
                            }
                            asc46 *= 10;
                        }
                        else
                        {
                            dtemp = (dtemp * 10) + (TEXT[j] - 48);
                        }
                    }
                }
                K[i] = mtemp * dtemp;
            }
            cnt = 0;
            for (int i = 0; i < K_M.rows; i++)
            {
                for (int j = 0; j < K_M.cols; j++)
                {
                    K_M.at<double>(i, j) = K[cnt];
                    cnt += 1;
                }
            }
            // cout<<K_M<<endl;
            // cout<<K[0]<<" "<<K[1]<<" "<<K[2]<<" "<<K[3]<<" "<<K[4]<<" "<<K[5]<<" "<<K[6]<<" "<<K[7]<<" "<<K[8]<<endl;
            for (int i = 0; i < 8; i++)
            {
                Calibration_Info >> temp;
            }
            for (int i = 0; i < 5; i++)
            {
                char TEXT[256];
                dtemp = 0;
                asc46 = 1;
                mtemp = 1;
                Calibration_Info >> temp;
                strcpy(TEXT, temp.c_str());
                for (std::string::size_type j = 0; j < temp.length(); j++)
                {
                    if (TEXT[j] == 45)
                    {
                        mtemp = -1;
                    }
                    if ((TEXT[j] >= 48 && TEXT[j] <= 57) || TEXT[j] == 46)
                    {
                        if (TEXT[j] == 46 || asc46 != 1)
                        {
                            if (TEXT[j] != 46)
                            {
                                dtemp += (TEXT[j] - 48) / asc46;
                            }
                            asc46 *= 10;
                        }
                        else
                        {
                            dtemp = (dtemp * 10) + (TEXT[j] - 48);
                        }
                    }
                }
                D[i] = mtemp * dtemp;
            }
            cnt = 0;
            for (int i = 0; i < D_M.rows; i++)
            {
                for (int j = 0; j < D_M.cols; j++)
                {
                    D_M.at<double>(i, j) = D[cnt];
                    cnt += 1;
                }
            }
            // cout<<D_M<<endl;
            // cout<<D[0]<<" "<<D[1]<<" "<<D[2]<<" "<<D[3]<<" "<<D[4]<<endl;
            for (int i = 0; i < 6; i++)
            {
                Calibration_Info >> temp;
            }
            for (int i = 0; i < 9; i++)
            {
                char TEXT[256];
                dtemp = 0;
                asc46 = 1;
                mtemp = 1;
                Calibration_Info >> temp;
                strcpy(TEXT, temp.c_str());
                for (std::string::size_type j = 0; j < temp.length(); j++)
                {
                    if (TEXT[j] == 45)
                    {
                        mtemp = -1;
                    }
                    if ((TEXT[j] >= 48 && TEXT[j] <= 57) || TEXT[j] == 46)
                    {
                        if (TEXT[j] == 46 || asc46 != 1)
                        {
                            if (TEXT[j] != 46)
                            {
                                dtemp += (TEXT[j] - 48) / asc46;
                            }
                            asc46 *= 10;
                        }
                        else
                        {
                            dtemp = (dtemp * 10) + (TEXT[j] - 48);
                        }
                    }
                }
                R[i] = mtemp * dtemp;
            }
            cnt = 0;
            for (int i = 0; i < R_M.rows; i++)
            {
                for (int j = 0; j < R_M.cols; j++)
                {
                    R_M.at<double>(i, j) = R[cnt];
                    cnt += 1;
                }
            }
            // cout<<R_M<<endl;
            // cout<<R[0]<<" "<<R[1]<<" "<<R[2]<<" "<<R[3]<<" "<<R[4]<<" "<<R[5]<<" "<<R[6]<<" "<<R[7]<<" "<<R[8]<<endl;
            for (int i = 0; i < 6; i++)
            {
                Calibration_Info >> temp;
            }
            for (int i = 0; i < 12; i++)
            {
                char TEXT[256];
                dtemp = 0;
                asc46 = 1;
                mtemp = 1;
                Calibration_Info >> temp;
                strcpy(TEXT, temp.c_str());
                for (std::string::size_type j = 0; j < temp.length(); j++)
                {
                    if (TEXT[j] == 45)
                    {
                        mtemp = -1;
                    }
                    if ((TEXT[j] >= 48 && TEXT[j] <= 57) || TEXT[j] == 46)
                    {
                        if (TEXT[j] == 46 || asc46 != 1)
                        {
                            if (TEXT[j] != 46)
                            {
                                dtemp += (TEXT[j] - 48) / asc46;
                            }
                            asc46 *= 10;
                        }
                        else
                        {
                            dtemp = (dtemp * 10) + (TEXT[j] - 48);
                        }
                    }
                }
                P[i] = mtemp * dtemp;
            }
            cnt = 0;
            for (int i = 0; i < P_M.rows; i++)
            {
                for (int j = 0; j < P_M.cols; j++)
                {
                    P_M.at<double>(i, j) = P[cnt];
                    cnt += 1;
                }
            }
            // cout<<P_M<<endl;
            // cout<<P[0]<<" "<<P[1]<<" "<<P[2]<<" "<<P[3]<<" "<<P[4]<<" "<<P[5]<<" "<<P[6]<<" "<<P[7]<<" "<<P[8]<<" "<<P[9]<<" "<<P[10]<<" "<<P[11]<<endl;
        }
        Calibration_Info.close();
        NEW_K_M = getOptimalNewCameraMatrix(K_M, D_M, Size(640, 480), 0.3, Size(640, 480), 0);
        cout << NEW_K_M << endl;

        new_focalLen = cv::Point2d(NEW_K_M.at<double>(0, 0), NEW_K_M.at<double>(1, 1));
        new_prncPt = cv::Point2d(NEW_K_M.at<double>(0, 2), NEW_K_M.at<double>(1, 2));
    }

private:
    float yaw = 0.0;

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    void imu_callback(const humanoid_interfaces::msg::ImuMsg::SharedPtr msg);

    void movo_publisher(float x);

    humanoid_interfaces::msg::MovoMsg movo_msg_;
    rclcpp::Publisher<humanoid_interfaces::msg::MovoMsg>::SharedPtr movo_publisher_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_subscription_;

    rclcpp::Subscription<humanoid_interfaces::msg::ImuMsg>::SharedPtr imu_subscription_;

    std::string image_topic;
};

#endif // MOVO_REALTIME_HPP
