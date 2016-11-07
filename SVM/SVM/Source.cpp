#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	char fileName[] = "a007.jpg";//自己隨便找張圖片測試即可
	IplImage *image;
	image = cvLoadImage(fileName, CV_LOAD_IMAGE_UNCHANGED);

	if (!image)
		cout << "找不到檔案!!!" << endl;
	else
	{
		cvShowImage("Test", image);
		cvWaitKey(0);
	}

	system("pause");
	return 0;
}