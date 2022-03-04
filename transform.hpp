//
//  transform.hpp
//  openCV
//
//  Created by Yifan Huang on 2022-02-04.
//

#ifndef transform_h
#define transform_h
#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <queue>
#include <cstring>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/core/utility.hpp"
#include "filter.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

//hough transform get votes
Mat* hough_Space(const Mat& src);
Mat* hough_Transform(const Mat& src, const int& lineThreshold);

void polarToCartesian(const int& rho, const int& theta, Point* p1, Point* p2);

Mat* gradiant_Square(const Mat& src);
Mat* gradiant_IxIy(const Mat& src1, const Mat& src2);

Mat* harri_Corner_Detect(const Mat& srcX, const Mat& srcY, double a);
#endif /* transform_h */
