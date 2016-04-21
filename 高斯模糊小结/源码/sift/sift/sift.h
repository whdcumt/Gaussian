//opencv2.2+vs2010
//zdd
//zddmail@gmail.com
//Copyright@All rights free for learning.

#ifndef SIFT_H
#define SIFT_H

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
using namespace cv;

void ConvertToGray(const Mat& src, Mat& dst);

void DownSample(const Mat& src, Mat& dst);

void UpSample(const Mat& src, Mat& dst);

void GaussianTemplateSmooth(const Mat &src, Mat &dst, double sigma);

void GaussianSmooth2D(const Mat &src, Mat &dst, double sigma);

void GaussianSmooth(const Mat &src, Mat &dst, double sigma);

#endif