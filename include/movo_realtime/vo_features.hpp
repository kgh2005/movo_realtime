#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator>  // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

// void featureTracking(Mat img_1, Mat img_2, vector<Point2f> &points1, vector<Point2f> &points2, vector<uchar> &status)
// {

//   // this function automatically gets rid of points for which tracking fails

//   vector<float> err;
//   Size winSize = Size(21, 21);
//   TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

//   calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

//   // getting rid of points for which the KLT tracking failed or those who have gone outside the frame
//   int indexCorrection = 0;
//   for (size_t i = 0; i < status.size(); i++)
//   {
//     Point2f pt = points2.at(i - indexCorrection);
//     if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0))
//     {
//       if ((pt.x < 0) || (pt.y < 0))
//       {
//         status.at(i) = 0;
//       }
//       points1.erase(points1.begin() + (i - indexCorrection));
//       points2.erase(points2.begin() + (i - indexCorrection));
//       indexCorrection++;
//     }
//   }
// }

// KLT 윈도우 크기와 피라미드 레벨 수
static const cv::Size  KLT_WIN_SIZE{21, 21};
static const int       KLT_PYR_LEVELS = 3;

// KLT 피처 트래킹 함수
// img1: 이전 프레임 이미지, img2: 현재 프레임 이미지
// pts1: 이전 프레임에서 추적할 포인트들, pts2: 현재 프레임에서 추적된 포인트들
// status: 각 포인트의 추적 성공 여부를 나타내는 상태 벡터
// 이 함수는 KLT 알고리즘을 사용하여 두 이미지 간의 피처 포인트를 추적하고,
// 유효하지 않은 포인트를 제거하며, RANSAC을 사용하여 아웃라이어를 제거합니다   
void featureTracking(const Mat &img1, const Mat &img2,
                     vector<Point2f> &pts1, vector<Point2f> &pts2,
                     vector<uchar> &status)
{
  // 1) 빈 포인트 처리
  if (pts1.empty()) {
    status.clear();
    return;
  }
  
  // 1) 피라미드 미리 생성
  vector<Mat> pyr1, pyr2;
  buildOpticalFlowPyramid(img1, pyr1, KLT_WIN_SIZE, KLT_PYR_LEVELS);
  buildOpticalFlowPyramid(img2, pyr2, KLT_WIN_SIZE, KLT_PYR_LEVELS);

  // 2) KLT 계산
  vector<float> err;
  TermCriteria tc(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
  calcOpticalFlowPyrLK(pyr1, pyr2, pts1, pts2, status, err,
                       KLT_WIN_SIZE, KLT_PYR_LEVELS, tc, 0, 1e-4);

  // 3) 유효 영역 벗어난 포인트 제거
  vector<Point2f> v1, v2;
  vector<uchar> vstat;
  for (size_t i = 0; i < status.size(); i++)
  {
    if (status[i] && pts2[i].inside(Rect(0, 0, img2.cols, img2.rows)))
    {
      v1.push_back(pts1[i]);
      v2.push_back(pts2[i]);
      vstat.push_back(1);
    }
  }
  pts1 = v1;
  pts2 = v2;
  status = vstat;

  // 4) RANSAC 으로 아웃라이어 제거 (기하학적 매칭 검증)
  if (pts1.size() >= 8)
  {
    Mat maskR;
    findFundamentalMat(pts1, pts2, FM_RANSAC, 1.0, 0.99, maskR);
    vector<Point2f> r1, r2;
    vector<uchar> rst;
    for (size_t i = 0; i < pts1.size(); i++)
    {
      if (maskR.at<uchar>(i))
      {
        r1.push_back(pts1[i]);
        r2.push_back(pts2[i]);
        rst.push_back(1);
      }
    }
    pts1 = r1;
    pts2 = r2;
    status = rst;
  }
}

void featureDetection(Mat img_1, vector<Point2f> &points1)
{ // uses FAST as of now, modify parameters as necessary
  // Mat gray;
  //  1) 한 번만 그레이로 변환
  //  if (img_1.channels() == 3)
  //  {
  //    cvtColor(img_1, gray, COLOR_BGR2GRAY);
  //  }
  //  else
  //  {
  //    gray = img_1;
  //  }

  // vector<KeyPoint> keypoints_1;
  // int fast_threshold = 20;
  // bool nonmaxSuppression = true;
  // FAST(gray, keypoints_1, fast_threshold, nonmaxSuppression);
  // KeyPoint::convert(keypoints_1, points1, vector<int>());

  // 2) 파라미터 설정
  const int fast_threshold = 20;
  const bool nonmaxSuppression = true;
  const int grid_x = 5;      // x축 격자 수
  const int grid_y = 4;      // y축 격자 수
  const int num_total = 200; // 최종 뽑을 최대 피처 개수
  int num_per_cell = num_total / (grid_x * grid_y) + 1;

  int cell_w = img_1.cols / grid_x; // 각 셀 너비
  int cell_h = img_1.rows / grid_y; // 각 셀 높이

  // 3) 격자마다 FAST 실행 후 응답값 기준 상위 셀당 num_per_cell개씩 수집
  vector<KeyPoint> all_kpts;
  all_kpts.reserve(num_total * 2);

  for (int gx = 0; gx < grid_x; gx++)
  {
    for (int gy = 0; gy < grid_y; gy++)
    {
      // 셀 영역 정의
      int x0 = gx * cell_w;
      int y0 = gy * cell_h;
      int w = (gx == grid_x - 1) ? (img_1.cols - x0) : cell_w;
      int h = (gy == grid_y - 1) ? (img_1.rows - y0) : cell_h;
      Rect roi(x0, y0, w, h);

      // FAST 검출
      vector<KeyPoint> kpts;
      FAST(img_1(roi), kpts, fast_threshold, nonmaxSuppression);

      // ROI 기준 → 전체 좌표로 보정
      for (auto &kp : kpts)
      {
        kp.pt.x += x0;
        kp.pt.y += y0;
      }

      // 응답값(response) 내림차순 정렬
      sort(kpts.begin(), kpts.end(),
           [](const KeyPoint &a, const KeyPoint &b)
           {
             return a.response > b.response;
           });

      // 셀당 상위 num_per_cell개만 취함
      for (int i = 0; i < (int)kpts.size() && i < num_per_cell; i++)
        all_kpts.push_back(kpts[i]);
    }
  }

  // 4) 전체 피처 중 다시 상위 num_total개로 컷오프
  sort(all_kpts.begin(), all_kpts.end(),
       [](const KeyPoint &a, const KeyPoint &b)
       {
         return a.response > b.response;
       });
  if ((int)all_kpts.size() > num_total)
    all_kpts.resize(num_total);

  // 5) KeyPoint → Point2f
  KeyPoint::convert(all_kpts, points1);

  // 6) 서브픽셀 정제
  if (!points1.empty())
  {
    vector<Point2f> refined = points1;
    TermCriteria termcrit(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03);
    cornerSubPix(img_1, refined, Size(5, 5), Size(-1, -1), termcrit);
    points1 = refined;
  }
}

// 이동 벡터 필터링 함수: Median Filter + Two-point RANSAC
void filterMotionVectors(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, std::vector<uchar> &status)
{
  std::vector<float> dx, dy;
  std::vector<cv::Point2f> filtered_points1, filtered_points2;

  for (size_t i = 0; i < status.size(); ++i)
  {
    if (status[i])
    {
      dx.push_back(points2[i].x - points1[i].x);
      dy.push_back(points2[i].y - points1[i].y);
    }
  }

  if (dx.empty())
    return;

  // Median 계산
  std::nth_element(dx.begin(), dx.begin() + dx.size() / 2, dx.end());
  std::nth_element(dy.begin(), dy.begin() + dy.size() / 2, dy.end());
  float median_dx = dx[dx.size() / 2];
  float median_dy = dy[dy.size() / 2];

  // Threshold
  float threshold = 5.0;

  for (size_t i = 0; i < status.size(); ++i)
  {
    if (!status[i])
      continue;
    float ddx = points2[i].x - points1[i].x - median_dx;
    float ddy = points2[i].y - points1[i].y - median_dy;
    float dist = std::sqrt(ddx * ddx + ddy * ddy);
    if (dist < threshold)
    {
      filtered_points1.push_back(points1[i]);
      filtered_points2.push_back(points2[i]);
    }
  }

  points1 = filtered_points1;
  points2 = filtered_points2;
}