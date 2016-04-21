//vs2010+opencv2.2
//zdd
//zddmail@gmail.com
//Copyright@All rights free for learning.

#include <windows.h>

#include <iostream>
using namespace std;

#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
using namespace cv;

#include "sift.h"


int main(int argc, char **argv)
{
	Mat src = imread("lena.jpg");

	Mat gray, dst;
	Mat small;
	ConvertToGray(src, gray);
	DownSample(gray, dst);
//	DownSample(dst, small);
	Mat up, up1;
//	UpSample(small, up);
//	Size sz(gray.size().width/2, gray.size().height/2);
//resize(src, small, sz);
//	resize(small, up1, small.size()*2);
//	imshow("up1", up1);
//	imshow("up", up);


	Mat gau, gau1;
	int start = GetTickCount();
	GaussianBlur(gray, gau, Size(0,0), 0.84089642); 
	cout << "GaussianBlur: "<< GetTickCount()-start << "ms"<<endl;
	imshow("gau", gau);

//	cvSmooth(gray, gau1, 2, 0,0, 0.0, 0.0);
//	GaussianBlur(gray, gau1, Size(0,0), 0.0); 
//	imshow("gau1", gau1);


	Mat gauss;
	start = GetTickCount();
	GaussianTemplateSmooth(gray, gauss, 0.84089642);
	cout << "GaussianTemplateSmooth: "<< GetTickCount()-start << "ms"<<endl;
	imshow("gauss", gauss);

//	imwrite("dst.jpg", dst);
//	imwrite("gauss.jpg", gauss);

	//0.84089642
	Mat gs;
	start = GetTickCount();
	GaussianSmooth(gray, gs, 0.84089642);
	cout << "GaussianSmooth: "<< GetTickCount()-start << "ms"<<endl;
	imshow("gs", gs);

//	imwrite("small.jpg", small);
//	imwrite("gs.jpg", gs);
	

	Mat g1;
	start = GetTickCount();
	GaussianSmooth2D(dst, g1, 10);
	cout << "GaussianSmooth2D: "<< GetTickCount()-start << "ms"<<endl;
	imshow("g1", g1);
	imwrite("g1.jpg",g1);

//	imwrite("g1.jpg", g1);

	if(src.empty() || gray.empty() || dst.empty())
		return -1;

//	imshow("src", src);
	imshow("gray", gray);
	imshow("dst", dst);

	waitKey();
	return 0;
}