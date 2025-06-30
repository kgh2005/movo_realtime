#include "../include/movo_realtime/movo_realtime.hpp"

#include "../include/movo_realtime/vo_features.hpp"

using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000

MovoRealtime::MovoRealtime() : Node("movo_realtime")
{
  image_topic = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");

  image_topic = this->get_parameter("image_topic").as_string();

  image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(image_topic, 10, std::bind(&MovoRealtime::image_callback, this, std::placeholders::_1));

  info_subscription_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera/camera_info", 10, [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg)
                                                                               {
                                                                                 K_M = Mat(3, 3, CV_64F, (void *)msg->k.data()).clone();
                                                                                 // if (!K_M.isContinuous()) {
                                                                                 //   K_M = K_M.clone();  // 연속된 메모리 확보
                                                                                 // }
                                                                                 // std::memcpy(K_M.data, msg->k.data(), 9 * sizeof(double));

                                                                                 D_M = Mat(1, 5, CV_64F, (void *)msg->d.data()).clone();
                                                                                 // std::memcpy(D_M.data, msg->d.data(), 5 * sizeof(double));

                                                                                 R_M = Mat(3, 3, CV_64F, (void *)msg->r.data()).clone();
                                                                                 // std::memcpy(R_M.data, msg->r.data(), 9 * sizeof(double));

                                                                                 P_M = Mat(3, 4, CV_64F, (void *)msg->p.data()).clone();
                                                                                 // std::memcpy(P_M.data, msg->p.data(), 12 * sizeof(double));

                                                                                 // K_M = cv::Mat(3, 3, CV_64F, (void *)msg->k.data());
                                                                                 // D_M = cv::Mat(1, 5, CV_64F, (void *)msg->d.data());
                                                                                 // R_M = cv::Mat(3, 3, CV_64F, (void *)msg->r.data());
                                                                                 // P_M = cv::Mat(3, 4, CV_64F, (void *)msg->p.data());
                                                                               });

  imu_subscription_ = this->create_subscription<humanoid_interfaces::msg::ImuMsg>(
      "Imu", 10,
      std::bind(&MovoRealtime::imu_callback, this, std::placeholders::_1));

  movo_publisher_ = this->create_publisher<humanoid_interfaces::msg::MovoMsg>("/movo_publish", 10);

  calibration_info();
}

MovoRealtime::~MovoRealtime()
{
  // 소멸자 구현
}

void MovoRealtime::imu_callback(const humanoid_interfaces::msg::ImuMsg::SharedPtr msg)
{
  yaw = msg->yaw;
  RCLCPP_INFO(this->get_logger(), "Received IMU yaw: %f", yaw);
  // You can store yaw for use in processing
}

// void MovoRealtime::movo_publisher(float x) {
//   movo_msg_.x = x * 20.0f;

//   movo_publisher_->publish(movo_msg_);
// }

void MovoRealtime::image_processing(const cv::Mat &img)
{
  static bool first = true;
  static bool second = false;
  static Mat traj = Mat::zeros(600, 600, CV_8UC3);
  static Point2f trajectory_pos(300, 300); // 시작 위치
  static Mat R_total = Mat::eye(3, 3, CV_64F);
  static Mat t_total = Mat::zeros(3, 1, CV_64F);

  Mat display_frame = img.clone();

  if (first && !second)
  {
    currImage_c = img.clone();
    second = true;
    first = false;
    return;
  }
  else if (second && !first)
  {
    prevImage = currImage_c.clone();
    currImage_c = img.clone();

    Mat img_1, img_2;
    cvtColor(prevImage, img_1, COLOR_BGR2GRAY);
    cvtColor(currImage_c, img_2, COLOR_BGR2GRAY);

    // points1.clear();
    // points2.clear();
    keypoints_1.clear();
    keypoints_2.clear();

    featureDetection(img_1, points1);

    // 시각화
    // Mat vis;
    // cvtColor(img_1, vis, COLOR_GRAY2BGR); // BGR이라면 clone()만 해도 됩니다.
    for (auto &pt : points1)
      circle(prevImage, pt, 3, Scalar(0, 255, 0), -1);
    imshow("Detected Features", prevImage);
    waitKey(1);

    vector<Point2f> prevFeatures, currFeatures;
    featureTracking(img_1, img_2, points1, points2, status);
    filterMotionVectors(points1, points2, status);
    if (points1.size() < 8 || points2.size() < 8)
    {
      RCLCPP_WARN(this->get_logger(), "Too few feature points. Skipping frame.");
      return;
    }

    E = findEssentialMat(points2, points1, focalLen.y, prncPt, RANSAC, 0.999, 1.0, mask);
    if (E.empty() || E.cols != 3 || E.rows != 3)
    {
      RCLCPP_WARN(this->get_logger(), "Invalid Essential Matrix. Skipping frame.");
      return;
    }

    int inliers = recoverPose(E, points2, points1, R, t, focalLen.y, prncPt, mask);
    // // ✅ 정지 상태 오인 방지
    // double t_norm = cv::norm(t);
    // if (t_norm < 0.01 || inliers < 30) {  // 거의 움직이지 않았다고 판단되면 업데이트 스킵
    //   RCLCPP_INFO(this->get_logger(), "Negligible motion detected (|t| = %.5f). Skipping frame.", t_norm);
    //   return;
    // }

    // 1) inliers 부족 체크
    if (inliers < 5)
    {
      RCLCPP_WARN(get_logger(),
                  "Too few inliers: %d (<10). Skipping frame.", inliers);
      return;
    }

    Mat prevPts(2, points1.size(), CV_64F), currPts(2, points2.size(), CV_64F);

    for (size_t i = 0; i < points1.size(); i++)
    { // this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
      prevPts.at<double>(0, i) = points1.at(i).x;
      prevPts.at<double>(1, i) = points1.at(i).y;

      currPts.at<double>(0, i) = points2.at(i).x;
      currPts.at<double>(1, i) = points2.at(i).y;
    }

    t_total = t_total + R_total * t * 0.3; // Scale factor 0.5 적용
    R_total = R * R_total;

    points1 = points2;

    // 궤적 업데이트
    trajectory_pos.x += (t_total.at<double>(0) * 2);
    trajectory_pos.y += (t_total.at<double>(2) * 2);

    // 1) recoverPose 로부터 얻은 이동 벡터 t = [tx, ty, tz]ᵀ
    double dx = t.at<double>(0);  // 왼쪽(+)/오른쪽(–) 방향 이동
    double dz = t.at<double>(2);  // 전진(+)/후진(–) 방향 이동

    // 2) 전/후진 판정 (z축)
    const double motion_th = 0.01;  // z축 임계값
    if (std::abs(dz) > motion_th) {
      if (dz > 0)
        RCLCPP_INFO(get_logger(), ">> 전진 detected (dz=%.5f)", dz);
      else
        RCLCPP_INFO(get_logger(), "<< 후진 detected (dz=%.5f)", dz);
    }

    // 3) 좌/우 판정 (x축)
    const double lateral_th = 0.01; // x축 임계값
    if (std::abs(dx) > lateral_th) {
      if (dx > 0)
        RCLCPP_INFO(get_logger(), ">> 우측 이동 detected (dx=%.5f)", dx);
      else
        RCLCPP_INFO(get_logger(), "<< 좌측 이동 detected (dx=%.5f)", dx);
    }

    // R_total 에서 yaw(수직축 회전) 추출
    // RPY 중 yaw = atan2(r21, r11)  (ROS 좌표계: X→전진, Y→왼쪽, Z→위쪽)
    // OpenCV 행렬 R_total: R_total.at<double>(row, col)
    double yaw_vo = std::atan2(R_total.at<double>(1, 0), R_total.at<double>(0, 0));

    // ROS 로그로 출력
    // RCLCPP_INFO(get_logger(),
    //             "VO Pose — x: %.3f, y: %.3f, yaw: %.3f deg",
    //             x_vo,
    //             y_vo,
    //             yaw_vo * 180.0 / M_PI);

    // movo_publisher(x_vo); 
    movo_msg_.x = dz * 10.0; // 전진/후진 이동량을 20배로 스케일링
    movo_msg_.y = dx * 10.0; // 좌/우 이동량을 20배로 스케일링
    movo_msg_.yaw = yaw_vo * 180.0 / M_PI; // yaw를 도 단위로 변환
    movo_publisher_->publish(movo_msg_);

    if (dz == 0)
    {
      movo_msg_.x = 0.0;
      movo_publisher_->publish(movo_msg_);
    }
    if (dx == 0)
    {
      movo_msg_.y = 0.0;
      movo_publisher_->publish(movo_msg_);
    }

    circle(traj, trajectory_pos, 1, CV_RGB(255, 0, 0), 2);

    // 현재 상태 표시
    putText(display_frame, format("Tracked Features: %d", (int)good_matches.size()), Point(10, 30), FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 255, 0), 1);
    putText(display_frame, format("Inliers: %d", inliers), Point(10, 50), FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 255, 0), 1);
    // 결과 표시
    Mat traj_display = traj.clone();
    rectangle(traj_display, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);
    putText(traj_display, "Blue - Camera Path", Point(10, 45), FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 1);

    imshow("Camera View", display_frame);
    imshow("Trajectory", traj_display);
  }
  else
  {
    prevImage = currImage.clone();
    currImage = img.clone();
    keypoints_2.clear();
    descriptors_2 = Mat();
    good_matches.clear();
  }

  char key = waitKey(1);
  if (key == 27)
  { // ESC
    rclcpp::shutdown();
  }
  else if (key == 'r')
  { // Reset trajectory
    traj = Mat::zeros(600, 600, CV_8UC3);
    trajectory_pos = Point2f(300, 300);
    R_total = Mat::eye(3, 3, CV_64F);
    t_total = Mat::zeros(3, 1, CV_64F);
  }
}

void MovoRealtime::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  static cv::Mat frame; // memory leak
  try
  {
    frame = cv_bridge::toCvShare(msg, "bgr8")->image;

    if (frame.empty())
    {
      RCLCPP_ERROR(this->get_logger(), "Received empty frame.");
      return;
    }

    if (cv::waitKey(10) == 27)
    {
      rclcpp::shutdown();
    }
  }
  catch (cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }

  focalLen = cv::Point2d(K_M.at<double>(0, 0), K_M.at<double>(1, 1));
  prncPt = cv::Point2d(K_M.at<double>(0, 2), K_M.at<double>(1, 2));

  image_processing(frame);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<MovoRealtime>());

  rclcpp::shutdown();
  return 0;
}
