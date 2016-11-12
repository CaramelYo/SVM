#include <iostream>
#include <vector>
#include <dirent.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <cmath>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

void GetFiles(string dir, vector<string> &files);

int main()
{
	string dir = string("./CSL/test");
	vector<string> files = vector<string>();
	int imgWidth = 227, imgHeight = 227, imgChannel = 3, imgWidthLimit = imgChannel * imgWidth;

	GetFiles(dir, files);
	/*
	//to output the file name
	int fileSize = files.size();
	for (int i = 0; i < fileSize; ++i)
	{
		cout << files[i] << endl;
	}
	*/

	vector<int> labels = vector<int>();
	vector<uchar*> trainingData = vector<uchar*>();
	int fileSize = files.size();

	//to get data & labels
	//to skip . & ..
	for (int i = 2; i < 3; ++i)
	{
		if (files[i][0] == 'a')
		{
			labels.push_back(1);
		}
		else
		{
			labels.push_back(-1);
		}

		Mat img = imread(dir + "/" + files[i], CV_LOAD_IMAGE_COLOR);
		//trainingData.push_back(img.data);

		cout << (int)img.data[0] << endl;
		cout << (int)img.ptr<uchar>(0, 0) << endl;
	}

	system("pause");


	int trainingDataSize = labels.size();

	//Mat trainingDataMat(trainingDataSize, imgWidth * imgHeight, CV_32FC1, trainingData);

	//cout << testDataSize << endl;

	/*
	for (int i = 0; i < fileSize; ++i)
	{
		cout <<labels[i] << endl;
	}
	*/

	//to create labels matrix
	//int *labelsArr = &labels[0];
	int labelsArr[5] = { labels[0], labels[1], labels[2], labels[3], labels[4] };
	//uchar* trainingDataArr = trainingData[0];
	uchar trainingDataArr[1][51529];

	/*
	for (int i = 0; i < 51529; ++i)
	{
		cout << (int)trainingData[0][i];
	}
	*/

	cout << (unsigned int)trainingData[0][0] << " " << (unsigned int)trainingData[0][1] << endl;

	//can the type change?
	//Mat labelsMat(trainingDataSize, 1, CV_32SC1, labelsArr);
	Mat labelsMat(5, 1, CV_32SC1, labelsArr);
	//Mat trainingDataMat(trainingDataSize, imgWidth * imgHeight, CV_32SC1, trainingDataArr);
	Mat trainingDataMat(5, 51529, CV_32SC1, trainingDataArr);
	system("pause");

	/*
	//to train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	cout << "go" << endl;
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);

	Mat img = imread("./CSL/test/a008.jpg");
	Mat sampleMat = Mat(1, imgWidth * imgHeight, CV_32SC1, img.data);
	float response = svm->predict(sampleMat);
	cout << response << endl;
	*/

	/*
	// Show the decision regions given by the SVM
	Mat image = Mat::zeros(500, 500, CV_8UC3);
	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255);

	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = svm->predict(sampleMat);

			if (response == 0)
				image.at<Vec3b>(i, j) = green;
			else if (response == 1)
				image.at<Vec3b>(i, j) = blue;
			else if (response == 2)
				image.at<Vec3b>(i, j) = red;
		}
	// Show the training data
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
	// Show support vectors
	thickness = 2;
	lineType = 8;
	Mat sv = svm->getUncompressedSupportVectors();
	for (int i = 0; i < sv.rows; ++i)
	{
		const float* v = sv.ptr<float>(i);
		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
	}
	imwrite("result.png", image);        // save the image
	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);
	*/


	/*
	for (int i = 0; i < testDataSize; ++i)
	{
		unsigned int *data = labelsMat.ptr<unsigned int>(i);
		cout << data[0] << endl;
	}
	*/
	
	//system("pause");

	/*
	Mat img = imread("./CSL/test/a007.jpg", CV_LOAD_IMAGE_COLOR);

	if (!img.data)
	{
		cout << "Reading img Failed";
		return 0;
	}

	namedWindow("Display Img", WINDOW_NORMAL);
	imshow("Display Img", img);
	waitKey(0);
	
	cout << img.cols << " " << img.rows << " " << img.channels() << endl;
	Mat trainingDataMat(trainingDataSize, imgWidth * imgHeight, CV_32FC1, img.data);
	//cout << img.ptr<Vec3b>(226, 110)<< endl;
	
	system("pause");
	*/
	/*
	// Data for visual representation
	int width = 227, height = 227;
	Mat image = Mat::zeros(height, width, CV_8UC3);
	// Set up training data
	int labels[4] = { 1, -1, -1, -1 };
	float trainingData[4][2] = { { 501, 10 },{ 255, 10 },{ 501, 255 },{ 10, 501 } };
	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
	Mat labelsMat(4, 1, CV_32SC1, labels);
	// Train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	// Show the decision regions given by the SVM
	Vec3b green(0, 255, 0), blue(255, 0, 0);
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = svm->predict(sampleMat);
			if (response == 1)
				image.at<Vec3b>(i, j) = green;
			else if (response == -1)
				image.at<Vec3b>(i, j) = blue;
		}
	// Show the training data
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
	// Show support vectors
	thickness = 2;
	lineType = 8;
	Mat sv = svm->getUncompressedSupportVectors();
	for (int i = 0; i < sv.rows; ++i)
	{
		const float* v = sv.ptr<float>(i);
		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
	}
	imwrite("result.png", image);        // save the image
	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);
	*/

	return 0;
}

void GetFiles(string dir, vector<string> &files)
{
	//to create directory pointer
	DIR *dp;
	struct dirent *dirp;

	//error detection
	if ((dp = opendir(dir.c_str())) == NULL)
	{
		cerr << "Error: " << errno << "opening" << dir << endl;
		return;
	}

	//to read all file name into files
	while ((dirp = readdir(dp)) != NULL)
		files.push_back(string(dirp->d_name));

	//to close directory pointer
	closedir(dp);
	return;
}