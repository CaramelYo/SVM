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

#define fileSize 102
#define imgHeight 227
#define imgWidth 227
#define imgWidthLimit imgWidth * 3
#define imgLimit imgHeight * imgWidth * 3

using namespace cv;
using namespace cv::ml;
using namespace std;

void GetFiles(string dir, string *files); 
PCA PCACompression(const Mat& pcaset, int maxComponents, const Mat& testset, Mat& compressed);

int main()
{
	//to get the training data
	string dir = string("./CSL/test");
	//vector<string> files = vector<string>();
	string files[fileSize];

	GetFiles(dir, files);

	/*
	//to output the file name
	int fileSize = files.size();
	for (int i = 0; i < fileSize; ++i)
	{
		cout << files[i] << endl;
	}
	*/

	//vector<int> labels = vector<int>();
	int labels[fileSize];
	//vector<uint8_t*> trainingData = vector<uint8_t*>();
	uint8_t *trainingData[fileSize];
	//int fileSize = files.size();
	uint8_t *a[fileSize];
	uint8_t *aa;
	a[20] = aa;
	cout << a[20] << endl;
	trainingData[102][0] = 1;
	cout << trainingData[102][0] << endl;

	//to get data & labels
	//to skip . & ..
	for (int i = 2; i < fileSize; ++i)
	{
		int j = i - 2;
		if (files[i][0] == 'a')
			labels[j] = 1;
		else
			labels[j] = -1;

		Mat img = imread(dir + "/" + files[i], CV_LOAD_IMAGE_COLOR);
		trainingData[j] = img.data;
	}

	waitKey(0);

	//image setting
	//int imgWidth = 227, imgHeight = 227, imgChannel = 3, imgLimit = imgHeight * imgWidth * imgChannel;
	//int trainingDataSize = labels.size();
	int trainingDataSize = fileSize - 2;

	//to get the image features by pca
	Mat trainingDataMat(trainingDataSize, imgHeight * imgWidth, CV_8UC3), compressedTrainingData;
	uint8_t *trainingDataMatPtr = trainingDataMat.data;

	//cout << "trainingDataSize = " << trainingDataSize << " training data size = " << trainingData.size() << endl;
	//CV_Assert(trainingDataSize == trainingData.size());
	
	for (int i = 0; i < trainingDataSize; ++i)
	{
		int index = i * imgWidthLimit;
		for (int j = 0; j < imgWidthLimit; ++j)
		{
			try
			{
				*(trainingDataMatPtr + index + j) = *(trainingData[i] + j);
			}
			catch (...)
			{
				//cerr << e.err << endl;
				//cout << e.err << endl;
				cout << "?";
			}
		}
	}


	//PCA pca = PCACompression(trainingData, 2, trainingData, compressedTrainingData);

	//Mat trainingDataMat(trainingDataSize, imgWidth * imgHeight, CV_32FC1);
	
	//Mat trainingDataMat(trainingDataSize, ;

	cout << "height = " << trainingDataMat.rows << " width = " << trainingDataMat.cols << endl;
	Mat test = Mat(trainingDataMat.row(0)).reshape(0, imgHeight);
	Mat test1 = imread("./CSL/test/a007.jpg");

	/*
	for (int i = 0; i < imgHeight; ++i)
	{
		for (int j = 0; j < imgWidth; ++j)
		{

		}
	}
	*/

	imshow("Image Test", test);

	imshow("Image Test1", test1);
	
	waitKey(0);
	return 0;

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
	//Mat trainingDataMat(5, 51529, CV_32SC1, trainingDataArr);
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

void GetFiles(string dir, string *files)
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
	//while ((dirp = readdir(dp)) != NULL)
	//	files.push_back(string(dirp->d_name));
	for (int i = 0; i < fileSize; ++i)
	{
		if ((dirp = readdir(dp)) != NULL)
			files[i] = string(dirp->d_name);
	}

	//to close directory pointer
	closedir(dp);
	return;
}

PCA PCACompression(const Mat& pcaset, int maxComponents,
	const Mat& testset, Mat& compressed)
{
	//pca(input data, Mat() means that PCA engine will compute the mean vector, thr form of eigenvector in matrix, the count of componet)
	PCA pca(pcaset, Mat(), PCA::DATA_AS_ROW, maxComponents);

	// if there is no test data, just return the computed basis, ready-to-use
	if (!testset.data)
		return pca;

	//to ensure if the cols of testset are equal to the cols of pcaset;
	CV_Assert(testset.cols == pcaset.cols);


	compressed.create(testset.rows, maxComponents, testset.type());
	//Mat reconstructed;???

	int rows = testset.rows;
	for (int i = 0; i < rows; i++)
	{
		Mat vec = testset.row(i), coeffs = compressed.row(i), reconstructed;
		// compress the vector, the result will be stored in the i-th row of the output matrix
		pca.project(vec, coeffs);
		// to reconstruct it
		//pca.backProject(coeffs, reconstructed);
		// to measure the error
		//printf("%d. diff = %g\n", i, norm(vec, reconstructed, NORM_L2));
	}
	return pca;
}