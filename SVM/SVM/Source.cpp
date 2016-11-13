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

#define fileSize 802
#define imgHeight 227
#define imgWidth 227
#define imgWidthLimit imgWidth * 3
#define imgLimit imgHeight * imgWidth
#define imgChLimit imgHeight * imgWidth * 3

using namespace cv;
using namespace cv::ml;
using namespace std;

void GetFiles(string dir, string *files); 
PCA PCACompression(const Mat& pcaset, int maxComponents, const Mat& testset, Mat& compressedPcaset, Mat& compressedTestset);

int main()
{
	//to get the training data
	string trainingDir = string("./CSL/training"), testDir = string("./CSL/test");
	//vector<string> files = vector<string>();
	string trainingFiles[fileSize], testFiles[fileSize];

	GetFiles(trainingDir, trainingFiles);
	GetFiles(testDir, testFiles);

	/*
	//to output the file name
	int fileSize = files.size();
	for (int i = 0; i < fileSize; ++i)
	{
		cout << files[i] << endl;
	}
	*/

	//vector<int> labels = vector<int>();
	int trainingLabels[fileSize - 2], testLabels[fileSize - 2];
	//vector<uint8_t*> trainingData = vector<uint8_t*>();

	uint8_t *trainingImage[fileSize - 2], *trainingData[fileSize - 2], *testImage[fileSize - 2], *testData[fileSize - 2];
	//int fileSize = files.size();

	/*
	uint8_t *a[fileSize - 2];
	//uint8_t *aa = a[0];
	
	Mat img1 = imread("./CSL/test/a007.jpg", CV_LOAD_IMAGE_COLOR);
	uint8_t *p = img1.data;
	a[0] = new uint8_t[imgLimit];

	for (int i = 0; i < imgLimit; ++i)
	{
		a[0][i] = p[i];
	}
	img1 = Mat(imgHeight, imgWidth, CV_8UC3);
	//cout << img2.type() << endl;
	//cvSetData(img1.data, a[0], img1.step);
	img1.data = a[0];
	imshow("YO",img1);
	waitKey(0);
	system("pause");
	return 0;
	*/

	//a[20] = aa;
	//cout << a[20] << endl;
	//trainingData[102][0] = 1;
	//cout << trainingData[102][0] << endl;

	//to get data & labels
	//to skip . & ..
	for (int i = 2; i < fileSize; ++i)
	{
		int j = i - 2;

		if (trainingFiles[i][0] == 'a')
			trainingLabels[j] = 1;
		else if (trainingFiles[i][0] == 'b')
			trainingLabels[j] = 2;
		else
			trainingLabels[j] = -1;

		if (testFiles[i][0] == 'a')
			testLabels[j] = 1; 
		else if (testFiles[i][0] == 'b')
			trainingLabels[j] = 2;
		else
			testLabels[j] = -1;


		Mat training = imread(trainingDir + "/" + trainingFiles[i], CV_LOAD_IMAGE_COLOR), test = imread(testDir + "/" + testFiles[i], CV_LOAD_IMAGE_COLOR);
		uint8_t *pTraining = training.data, *pTest = test.data;

		trainingImage[j] = new uint8_t[imgChLimit];
		trainingData[j] = new uint8_t[imgChLimit];
		testImage[j] = new uint8_t[imgChLimit];
		testData[j] = new uint8_t[imgLimit];
		
		for (int k = 0; k < imgChLimit; ++k)
		{
			trainingImage[j][k] = pTraining[k];
			testImage[j][k] = pTest[k];
		}

		cvtColor(training, training, CV_RGB2GRAY);
		cvtColor(test, test, CV_RGB2GRAY);
		pTraining = training.data;
		pTest = test.data;

		for (int k = 0; k < imgLimit; ++k)
		{
			trainingData[j][k] = pTraining[k];
			testData[j][k] = pTest[k];
		}
	}

	//image setting
	//int imgWidth = 227, imgHeight = 227, imgChannel = 3, imgLimit = imgHeight * imgWidth * imgChannel;
	//int trainingDataSize = labels.size();
	int trainingDataSize = fileSize - 2, testDataSize = fileSize - 2;

	//to change the form of data from uint8_t* array to Mat
	Mat trainingDataMat(trainingDataSize, imgLimit, CV_32FC1), testDataMat(testDataSize, imgLimit, CV_32FC1);
	float *trainingDataMatPtr = (float*)trainingDataMat.data, *testDataMatPtr = (float*)testDataMat.data;

	for (int i = 0; i < trainingDataSize; ++i)
	{
		int index = i * imgLimit;
		for (int j = 0; j < imgLimit; ++j)
			trainingDataMatPtr[index + j] = (float)trainingData[i][j];
	}

	for (int i = 0; i < testDataSize; ++i)
	{
		int index = i * imgLimit;
		for (int j = 0; j < imgLimit; ++j)
			testDataMatPtr[index + j] = (float)testData[i][j];
	}

	//to change the form of labels from int array to Mat
	//Mat trainingLabelsMat(trainingDataSize, 1, CV_8SC1, trainingLabels), testLabelsMat(testDataSize, 1, CV_8SC1, testLabels);
	Mat trainingLabelsMat(trainingDataSize, 1, CV_32S), testLabelsMat(testDataSize, 1, CV_32S);
	int *trainingLabelsMatPtr = (int*)trainingLabelsMat.data, *testLabelsMatPtr = (int*)testLabelsMat.data;

	for (int i = 0; i < trainingDataSize; ++i)
		trainingLabelsMatPtr[i] = trainingLabels[i];

	for (int i = 0; i < testDataSize; ++i)
		testLabelsMatPtr[i] = testLabels[i];

	/*
	Mat test = Mat(trainingDataMat.row(99)).reshape(0, imgHeight);
	//Mat test1 = imread("./CSL/training/a113.jpg");
	Mat test1 = Mat(trainingDataMat.row(45)).reshape(0, imgHeight);
	Mat test2 = Mat(trainingDataMat.row(456)).reshape(0, imgHeight);
	Mat test3 = Mat(testDataMat.row(45)).reshape(0, imgHeight);
	Mat test4 = Mat(testDataMat.row(356)).reshape(0, imgHeight);

	imshow("Test", test);
	imshow("Test1", test1);
	imshow("Test2", test2);
	imshow("Test3", test3);
	imshow("Test4", test4);
	waitKey(0);
	*/

	//to get the image features by pca
	Mat compressedTrainingDataMat, compressedTestDataMat;
	int components = 2;
	PCA pca = PCACompression(trainingDataMat, components, testDataMat, compressedTrainingDataMat, compressedTestDataMat);
	normalize(compressedTrainingDataMat, compressedTrainingDataMat, 0.0, 500.0, NORM_MINMAX);
	normalize(compressedTestDataMat, compressedTestDataMat, 0.0, 500.0, NORM_MINMAX);

	/*
	double *p = (double*)reconstructedTestData.data;
	Mat test(imgHeight, imgWidth, CV_8UC1);
	uint8_t *pp = test.data;
	cout << "int = " << (int)p[255] << " uint8_t" << (uint8_t)p[255] << endl;

	for (int i = 0; i < imgLimit; ++i)
	{
		pp[i] = (int)p[i];
	}

	for (int i = 0; i < imgLimit; ++i)
	{
		if (i % imgHeight == 0)
			cout << endl;
		
		cout << (int)pp[i] << " ";
	}
	imshow("Re test", test);
	waitKey(0);
	*/


	/*
	cout << "height = " << compressedTrainingData.rows << " width = " << compressedTrainingData.cols << " type = " << compressedTrainingData.type() << endl;
	//uint8_t *p = compressedTrainingData.data;

	for (int i = 0; i < testDataSize; ++i)
	{
		int j = i * 3;
		cout << compressedTrainingData.at<double>(i, 0) << "," << compressedTrainingData.at<double>(i, 1) << "," << compressedTrainingData.at<double>(i, 2) << endl;
	}
	*/

	//cout << testDataSize << endl;

	/*
	for (int i = 0; i < fileSize; ++i)
	{
		cout <<labels[i] << endl;
	}
	*/

	//to create labels matrix
	//int *labelsArr = &labels[0];
	//int labelsArr[5] = { labels[0], labels[1], labels[2], labels[3], labels[4] };
	//uchar* trainingDataArr = trainingData[0];
	//uchar trainingDataArr[1][51529];

	/*
	for (int i = 0; i < 51529; ++i)
	{
		cout << (int)trainingData[0][i];
	}
	*/

	//cout << (unsigned int)trainingData[0][0] << " " << (unsigned int)trainingData[0][1] << endl;

	//can the type change?
	//Mat labelsMat(trainingDataSize, 1, CV_32SC1, labelsArr);
	//Mat labelsMat(5, 1, CV_32SC1, labelsArr);
	//Mat trainingDataMat(trainingDataSize, imgWidth * imgHeight, CV_32SC1, trainingDataArr);
	//Mat trainingDataMat(5, 51529, CV_32SC1, trainingDataArr);
	//system("pause");

	
	//to train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	cout << "go" << endl;
	//cout << "data row = " << compressedTrainingDataMat.rows << " data col = " << compressedTrainingDataMat.cols << " labels row = " << trainingLabelsMat.rows << " labels col = " << trainingLabelsMat.cols << endl;
	svm->train(compressedTrainingDataMat, ROW_SAMPLE, trainingLabelsMat);

	//Mat img = imread("./CSL/test/a008.jpg");
	//Mat sampleMat = Mat(1, imgWidth * imgHeight, CV_32SC1, img.data);
	//Mat testMat8(1, components, CV_8UC1), testMat64(1, components, CV_32FC1);
	Mat testMat64(1, components, CV_32FC1);
	//uint8_t *p8 = testMat8.data, *p = compressedTestDataMat.data;
	float *p64 = (float*)testMat64.data, *compressedTestDataMatPtr = (float*)compressedTestDataMat.data;

	for (int i = 0; i < components; ++i)
	{
		//p8[i] = p[i];
		p64[i] = compressedTestDataMatPtr[i];
	}

	//cout << "testMat8 = " << (int)p8[0] << " " << (int)p8[1] << endl;
	//cout << "testMat64 = " << p64[0] << " " << p64[1] << endl;
	
	//cout << "test8 = " << svm->predict(testMat8) << " test64 = " << svm->predict(testMat64) << endl;
	//float response = svm->predict(testMat64);
	//cout << "test64 = " << response << endl;

	
	// Show the decision regions given by the SVM
	int resultHeight = 500, resultWidth = 500;
	Mat image = Mat::zeros(resultHeight, resultWidth, CV_8UC3);
	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255);

	for (int i = 0; i < resultHeight; ++i)
		for (int j = 0; j < resultWidth; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = svm->predict(sampleMat);

			if (response == 1)
				image.at<Vec3b>(i, j) = green;
			else if (response == 2)
				image.at<Vec3b>(i, j) = red;
			else if (response == -1)
				image.at<Vec3b>(i, j) = blue;

			/*
			if (response == 0)
				image.at<Vec3b>(i, j) = green;
			else if (response == 1)
				image.at<Vec3b>(i, j) = blue;
			else if (response == 2)
				image.at<Vec3b>(i, j) = red;
			*/
		}
	// Show the training data
	int thickness = -1;
	int lineType = 8;
	//circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	//circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	//circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	//circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
	int testDataSize2 = testDataSize * 2;
	for (int i = 0; i < testDataSize2; i+=2)
	{
		//cout << compressedTestDataMatPtr[i] << " " << compressedTestDataMatPtr[i + 1] << endl;
		Mat sampleMat = (Mat_<float>(1, 2) << compressedTestDataMatPtr[i], compressedTestDataMatPtr[i + 1]);
		float response = svm->predict(sampleMat);
		Scalar s;

		if (response == 1)
			s = Scalar(255, 255, 0);
		else if (response == 2)
			s = Scalar(0, 255, 255);
		else if (response == -1)
			s = Scalar(255, 0, 255);

		circle(image, Point(compressedTestDataMatPtr[i], compressedTestDataMatPtr[i + 1]), 1, s, thickness, lineType);
	}
	
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

//PCA PCACompression(const Mat& pcaset, int maxComponents, const Mat& testset, Mat& compressedPcaset, Mat& compressedTestset, Mat& reconstructedTestset)
PCA PCACompression(const Mat& pcaset, int maxComponents, const Mat& testset, Mat& compressedPcaset, Mat& compressedTestset)
{
	//pca(input data, Mat() means that PCA engine will compute the mean vector, thr form of eigenvector in matrix, the count of componet)
	PCA pca(pcaset, Mat(), PCA::DATA_AS_ROW, maxComponents);

	// if there is no test data, just return the computed basis, ready-to-use
	if (!testset.data)
		return pca;

	//to ensure if the cols of testset are equal to the cols of pcaset;
	CV_Assert(testset.cols == pcaset.cols);

	compressedPcaset.create(pcaset.rows, maxComponents, CV_32FC1);
	compressedTestset.create(testset.rows, maxComponents, CV_32FC1);
	//reconstructedTestset.create(testset.rows, testset.cols, CV_64FC1);

	//to get the new training data
	int rows = pcaset.rows;
	for (int i = 0; i < rows; ++i)
	{
		Mat vec = pcaset.row(i), coeffs = compressedPcaset.row(i);
		pca.project(vec, coeffs);
	}

	//to get the new test data
	rows = testset.rows;
	for (int i = 0; i < rows; i++)
	{
		//Mat vec = testset.row(i), coeffs = compressedTestset.row(i), re = reconstructedTestset.row(i);
		Mat vec = testset.row(i), coeffs = compressedTestset.row(i);
		// compress the vector, the result will be stored in the i-th row of the output matrix
		pca.project(vec, coeffs);
		// to reconstruct it
		//normalize(pca.backProject(coeffs).row(0), re, 0, 255, NORM_MINMAX);
		//normalize()
		// to measure the error
		//cout << i << ". diff = " << norm(vec, re, NORM_L2) << endl;
	}

	return pca;
}