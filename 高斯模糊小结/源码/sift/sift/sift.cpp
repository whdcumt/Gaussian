//vs2010+opencv2.2
//zdd
//zddmail@gmail.com
//Copyright@All rights free for learning.

#include "sift.h"

#include <fstream>
#include <iostream>
using namespace std;

//转换为灰度图像
void ConvertToGray(const Mat& src, Mat& dst)
{
	Size size = src.size();
	if(dst.empty())
		dst.create(size, CV_8U);

	uchar* srcData = src.data;
	uchar* dstData = dst.data;

	for(int j = 0; j < src.cols; j++)
	{
		for(int i = 0; i < src.rows; i++)
		{

			int b = *(srcData + src.step * i + src.channels() * j + 0);
			int g = *(srcData + src.step * i + src.channels() * j + 1);
			int r = *(srcData + src.step * i + src.channels() * j + 2);
			
			*(dstData + dst.step * i + dst.channels() * j) = (b + g + r)/3;
		}
	}
}

//隔点采样
void DownSample(const Mat& src, Mat& dst)
{
	if(src.channels() != 1)
		return;
	dst.create(src.rows/2, src.cols/2, src.type());

	uchar* srcData = src.data;
	uchar* dstData = dst.data;


	int m = 0, n = 0;
	for(int j = 0; j < src.cols; j+=2, n++)
	{
		m = 0;
		for(int i = 0; i < src.rows; i+=2, m++)
		{
			int sample = *(srcData + src.step * i + src.channels() * j);
			*(dstData + dst.step * m + dst.channels() * n) = sample;
		}
	}

}

//线性插值放大
void UpSample(const Mat &src, Mat &dst)
{
	if(src.channels() != 1)
		return;
	dst.create(src.rows*2, src.cols*2, src.type());

	uchar* srcData = src.data;
	uchar* dstData = dst.data;


	int m = 0, n = 0;
	for(int j = 0; j < src.cols; j++, n+=2)
	{
		m = 0;
		for(int i = 0; i < src.rows; i++, m+=2)
		{
			int sample = *(srcData + src.step * i + src.channels() * j);
			*(dstData + dst.step * m + dst.channels() * n) = sample;
			
			int rs = *(srcData + src.step * (i) + src.channels()*j)+(*(srcData + src.step * (i+1) + src.channels()*j));
			*(dstData + dst.step * (m+1) + dst.channels() * n) = cvRound(rs/2);
			int cs = *(srcData + src.step * i + src.channels()*(j))+(*(srcData + src.step * i + src.channels()*(j+1)));
			*(dstData + dst.step * m + dst.channels() * (n+1)) = cvRound(cs/2);

			int center = (*(srcData + src.step * (i+1) + src.channels() * j))
						+ (*(srcData + src.step * i + src.channels() * j))
						+ (*(srcData + src.step * (i+1) + src.channels() * (j+1)))
						+ (*(srcData + src.step * i + src.channels() * (j+1)));

			*(dstData + dst.step * (m+1) + dst.channels() * (n+1)) = cvRound(center/4);

		}

	}

	if(dst.rows < 2 || dst.cols < 2)
		return;
	for(int k = dst.rows-1; k >=0; k--)
	{
		*(dstData + dst.step *(k) + dst.channels()*(dst.cols-1))=*(dstData + dst.step *(k) + dst.channels()*(dst.cols-2));
	}
	for(int k = dst.cols-1; k >=0; k--)
	{
		*(dstData + dst.step *(dst.rows-1) + dst.channels()*(k))=*(dstData + dst.step *(dst.rows-2) + dst.channels()*(k)); 
	}

}

//高斯平滑
//未使用sigma，边缘无处理
void GaussianTemplateSmooth(const Mat &src, Mat &dst, double sigma)
{
	//高斯模板(7*7)，sigma = 0.84089642，归一化后得到
	static const double gaussianTemplate[7][7] = 
	{
		{0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067},
		{0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
		{0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
		{0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771},
		{0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
		{0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
		{0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067}
	};

	dst.create(src.size(), src.type());
	uchar* srcData = src.data;
	uchar* dstData = dst.data;

	for(int j = 0; j < src.cols-7; j++)
	{
		for(int i = 0; i < src.rows-7; i++)
		{
			double acc = 0;
			double accb = 0, accg = 0, accr = 0; 
			for(int m = 0; m < 7; m++)
			{
				for(int n = 0; n < 7; n++)
				{
					if(src.channels() == 1)
						acc += *(srcData + src.step * (i+n) + src.channels() * (j+m)) * gaussianTemplate[m][n];
					else
					{
						accb += *(srcData + src.step * (i+n) + src.channels() * (j+m) + 0) * gaussianTemplate[m][n];
						accg += *(srcData + src.step * (i+n) + src.channels() * (j+m) + 1) * gaussianTemplate[m][n];
						accr += *(srcData + src.step * (i+n) + src.channels() * (j+m) + 2) * gaussianTemplate[m][n];
					}
				}
			}
			if(src.channels() == 1)
				*(dstData + dst.step * (i+3) + dst.channels() * (j+3))=(int)acc;
			else
			{
				*(dstData + dst.step * (i+3) + dst.channels() * (j+3) + 0)=(int)accb;
				*(dstData + dst.step * (i+3) + dst.channels() * (j+3) + 1)=(int)accg;
				*(dstData + dst.step * (i+3) + dst.channels() * (j+3) + 2)=(int)accr;
			}
		}
	}
	
}


void GaussianSmooth2D(const Mat &src, Mat &dst, double sigma)
{
	if(src.channels() != 1)
		return;

	//确保sigma为正数 
	sigma = sigma > 0 ? sigma : 0;
	//高斯核矩阵的大小为(6*sigma+1)*(6*sigma+1)
	//ksize为奇数
	int ksize = cvRound(sigma * 3) * 2 + 1;
	
//cout << "ksize=" <<ksize<<endl;
//	dst.create(src.size(), src.type());
	if(ksize == 1)
	{
		src.copyTo(dst);	
		return;
	}

	dst.create(src.size(), src.type());

	//计算高斯核矩阵
	double *kernel = new double[ksize*ksize];

	double scale = -0.5/(sigma*sigma);
	const double PI = 3.141592653;
	double cons = -scale/PI;

	double sum = 0;

	for(int i = 0; i < ksize; i++)
	{
		for(int j = 0; j < ksize; j++)
		{
			int x = i-(ksize-1)/2;
			int y = j-(ksize-1)/2;
			kernel[i*ksize + j] = cons * exp(scale * (x*x + y*y));

			sum += kernel[i*ksize+j];
//			cout << " " << kernel[i*ksize + j];
		}
//		cout <<endl;
	}
	//归一化
	for(int i = ksize*ksize-1; i >=0; i--)
	{
		*(kernel+i) /= sum;
	}
/*
	ofstream out("output.txt");
	for(int i = 0; i < ksize; i++)
	{
		for(int j = 0; j < ksize; j++)
		{
	//		cout << " " << kernel[i*ksize + j];
			out << " " << kernel[i*ksize + j];
		}
	//	cout <<endl;
		out <<endl;
	}
*/
	uchar* srcData = src.data;
	uchar* dstData = dst.data;

	int center = (ksize-1) /2;
	//图像卷积运算,处理边缘
	for(int j = 0; j < src.cols; j++)
	{
		for(int i = 0; i < src.rows; i++)
		{
			double acc = 0;

			for(int m = -center, c = 0; m <= center; m++, c++)
			{
				for(int n = -center, r = 0; n <= center; n++, r++)
				{
					if((i+n) >=0 && (i+n) < src.rows && (j+m) >=0 && (j+m) < src.cols)
					{
						
						acc += *(srcData + src.step * (i+n) + src.channels() * (j+m)) * kernel[r*ksize+c]; 
				
					}
				}
			}


			*(dstData + dst.step * (i) + (j)) = (int)acc;
		}
	}

/*
	//图像卷积运算，无边缘处理
	for(int j = 0; j < src.cols-ksize; j++)
	{
		for(int i = 0; i < src.rows-ksize; i++)
		{
			double acc = 0;

			for(int m = 0; m < ksize; m++)
			{
				for(int n = 0; n < ksize; n++)
				{
					acc += *(srcData + src.step * (i+n) + src.channels() * (j+m)) * kernel[m*ksize+n]; 
				}
			}

		
			*(dstData + dst.step * (i + (ksize - 1)/2) + (j + (ksize -1)/2)) = (int)acc;
		}
	}
*/
	//模板边缘用原象素填充
/*
	for(int j = 0; j < src.cols; j++)
	{
		for(int i = src.rows - ksize; i < src.rows; i++)
		{
			*(dstData + dst.step * i + j) = *(srcData + src.step * i + j);
			*(dstData + dst.step * j + i) = *(srcData + src.step * j + i);
		}

		for(int i = 0; i < ksize; i++)
		{
			*(dstData + dst.step * i + j) = *(srcData + src.step * i + j); 
			*(dstData + dst.step * j + i) = *(srcData + src.step * j + i);
		}
	}
*/
	delete []kernel;
}

void GaussianSmooth(const Mat &src, Mat &dst, double sigma)
{
	if(src.channels() != 1 && src.channels() != 3)
		return;

	//
	sigma = sigma > 0 ? sigma : -sigma;
	//高斯核矩阵的大小为(6*sigma+1)*(6*sigma+1)
	//ksize为奇数
	int ksize = ceil(sigma * 3) * 2 + 1;

	//cout << "ksize=" <<ksize<<endl;
	//	dst.create(src.size(), src.type());
	if(ksize == 1)
	{
		src.copyTo(dst);	
		return;
	}

	//计算一维高斯核
	double *kernel = new double[ksize];

	double scale = -0.5/(sigma*sigma);
	const double PI = 3.141592653;
	double cons = 1/sqrt(-scale / PI);

	double sum = 0;
	int kcenter = ksize/2;
	int i = 0, j = 0;
	for(i = 0; i < ksize; i++)
	{
		int x = i - kcenter;
		*(kernel+i) = cons * exp(x * x * scale);//一维高斯函数
		sum += *(kernel+i);

//		cout << " " << *(kernel+i);
	}
//	cout << endl;
	//归一化,确保高斯权值在[0,1]之间
	for(i = 0; i < ksize; i++)
	{
		*(kernel+i) /= sum;
//		cout << " " << *(kernel+i);
	}
//	cout << endl;

	dst.create(src.size(), src.type());
	Mat temp;
	temp.create(src.size(), src.type());

	uchar* srcData = src.data;
	uchar* dstData = dst.data;
	uchar* tempData = temp.data;

	//x方向一维高斯模糊
	for(int y = 0; y < src.rows; y++)
	{
		for(int x = 0; x < src.cols; x++)
		{
			double mul = 0;
			sum = 0;
			double bmul = 0, gmul = 0, rmul = 0;
			for(i = -kcenter; i <= kcenter; i++)
			{
				if((x+i) >= 0 && (x+i) < src.cols)
				{
					if(src.channels() == 1)
					{
						mul += *(srcData+y*src.step+(x+i))*(*(kernel+kcenter+i));
					}
					else 
					{
						bmul += *(srcData+y*src.step+(x+i)*src.channels() + 0)*(*(kernel+kcenter+i));
						gmul += *(srcData+y*src.step+(x+i)*src.channels() + 1)*(*(kernel+kcenter+i));
						rmul += *(srcData+y*src.step+(x+i)*src.channels() + 2)*(*(kernel+kcenter+i));
					}
					sum += (*(kernel+kcenter+i));
				}
			}
			if(src.channels() == 1)
			{
				*(tempData+y*temp.step+x) = mul/sum;
			}
			else
			{
				*(tempData+y*temp.step+x*temp.channels()+0) = bmul/sum;
				*(tempData+y*temp.step+x*temp.channels()+1) = gmul/sum;
				*(tempData+y*temp.step+x*temp.channels()+2) = rmul/sum;
			}
		}
	}

	
	//y方向一维高斯模糊
	for(int x = 0; x < temp.cols; x++)
	{
		for(int y = 0; y < temp.rows; y++)
		{
			double mul = 0;
			sum = 0;
			double bmul = 0, gmul = 0, rmul = 0;
			for(i = -kcenter; i <= kcenter; i++)
			{
				if((y+i) >= 0 && (y+i) < temp.rows)
				{
					if(temp.channels() == 1)
					{
						mul += *(tempData+(y+i)*temp.step+x)*(*(kernel+kcenter+i));
					}
					else
					{
						bmul += *(tempData+(y+i)*temp.step+x*temp.channels() + 0)*(*(kernel+kcenter+i));
						gmul += *(tempData+(y+i)*temp.step+x*temp.channels() + 1)*(*(kernel+kcenter+i));
						rmul += *(tempData+(y+i)*temp.step+x*temp.channels() + 2)*(*(kernel+kcenter+i));
					}
					sum += (*(kernel+kcenter+i));
				}
			}
			if(temp.channels() == 1)
			{
				*(dstData+y*dst.step+x) = mul/sum;
			}
			else
			{
				*(dstData+y*dst.step+x*dst.channels()+0) = bmul/sum;
				*(dstData+y*dst.step+x*dst.channels()+1) = gmul/sum;
				*(dstData+y*dst.step+x*dst.channels()+2) = rmul/sum;
			}
		
		}
	}
	
	delete[] kernel;
}