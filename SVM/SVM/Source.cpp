#include <iostream>
#include <vector>
#include <dirent.h>
#include <string>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <cmath>

#include <opencv2/opencv.hpp>

//7048 1302
#define trainingFileSize 7048
#define testFileSize 1302
#define imgHeight 227
#define imgWidth 227
#define imgWidthLimit imgWidth * 3
#define imgLimit imgHeight * imgWidth
#define imgChLimit imgHeight * imgWidth * 3

using namespace cv;
using namespace cv::ml;
using namespace std;

void GetFiles(string dir, string *files, int fileSize); 
PCA PCACompression(const Mat& pcaset, int maxComponents, Mat& compressedPcaset);

int main()
{
	/*
	fstream fp;
	//fp.open("pca.txt", ios::out);
	fp.open("pca.txt", ios::in);
	if (!fp)
	{
		cout << "Fail to open file" << endl;
		return 0;
	}

	
	Mat test0(50, 50, CV_32FC1);
	float *p = (float*)test0.data;
	
	//to write
	for (int i = 0; i < 50; ++i)
	{
		float a = i;
		int index = i * 50;
		for (int j = 0; j < 50; ++j)
		{
			a += j * 0.01;
			p[index + j] = a;
		}
	}

	for (int i = 0; i < 50; ++i)
	{
		int index = i * 50;
		for (int j = 0; j < 50; ++j)
		{
			fp << p[index + j]  << " " << endl;
		}
	}
	
	//to read
	for (int i = 0; i < 50; ++i)
	{
		int index = i * 50;
		for (int j = 0; j < 50; ++j)
		{
			fp >> p[index + j];
		}
	}

	fp.close();
	
	for (int i = 0; i < 50; ++i)
	{
		int index = i * 50;
		for (int j = 0; j < 50; ++j)
		{
			cout << p[index + j] << " ";
		}

		cout << endl;
	}

	system("pause");

	return 0;
	*/

	//to get the training data
	string dir = string("./CSL/training");
	string *files = new string[trainingFileSize];
	GetFiles(dir, files, trainingFileSize);

	//the type of labels must be int (CV_32S)  (?)
	int *trainingLabels = new int [trainingFileSize - 2];
	uint8_t *trainingData[trainingFileSize - 2];
	int trainingDataSize = trainingFileSize - 2;
	
	//is the original image needed?
	//uint8_t *trainingImage[trainingFileSize - 2], *testImage[trainingFileSize - 2];

	//to get training data & labels
	//to skip . & ..
	for (int i = 2; i < trainingFileSize; ++i)
	{
		int j = i - 2;

		trainingLabels[j] = files[i][0] - 96;

		//is the original image needed?
		//Mat training = imread(trainingDir + "/" + trainingFiles[i], CV_LOAD_IMAGE_COLOR), test = imread(testDir + "/" + testFiles[i], CV_LOAD_IMAGE_COLOR);
		
		//to use grayscale
		Mat training = imread(dir + "/" + files[i], CV_LOAD_IMAGE_GRAYSCALE);
		//uint8_t *pTraining = training.data, *pTest = test.data;
		uint8_t *p = training.data;

		//trainingImage[j] = new uint8_t[imgChLimit];
		//to allocate new space   **important
		trainingData[j] = new uint8_t[imgLimit];
		/*
		for (int k = 0; k < imgChLimit; ++k)
		{
			trainingImage[j][k] = pTraining[k];
			testImage[j][k] = pTest[k];
		}

		cvtColor(training, training, CV_RGB2GRAY);
		cvtColor(test, test, CV_RGB2GRAY);
		pTraining = training.data;
		pTest = test.data;
		*/

		for (int k = 0; k < imgLimit; ++k)
			trainingData[j][k] = p[k];

		training.release();
	}

	//to change the form of data from uint8_t* array to CV_32FC1 Mat
	//for pca and svm, the type of data matrix must be CV_32FC1  **important
	Mat trainingDataMat(trainingDataSize, imgLimit, CV_32FC1);
	float *trainingDataMatPtr = (float*)trainingDataMat.data;

	for (int i = 0; i < trainingDataSize; ++i)
	{
		int index = i * imgLimit;
		for (int j = 0; j < imgLimit; ++j)
			trainingDataMatPtr[index + j] = (float)trainingData[i][j];
	}

	for(int i = 0; i < trainingDataSize; ++i)
		delete[] trainingData[i];

	
	//to get the features by pca
	Mat compressedTrainingDataMat;
	int components = 26;
	PCA pca = PCACompression(trainingDataMat, components, compressedTrainingDataMat);
	cout << "YOP" << endl;
	trainingDataMat.release();
	
	//return 0;

	//to get test data
	dir = string("./CSL/test");
	
	delete[] files;
	files = new string[testFileSize];
	GetFiles(dir, files, testFileSize);

	//the type of labels must be int (CV_32S)  (?)
	int *testLabels = new int[testFileSize - 2];
	uint8_t *testData[testFileSize - 2];
	int testDataSize = testFileSize - 2;

	//to get test data & labels
	for (int i = 2; i < testFileSize; ++i)
	{
		int j = i - 2;

		testLabels[j] = files[i][0] - 96;

		//to use grayscale
		Mat test = imread(dir + "/" + files[i], CV_LOAD_IMAGE_GRAYSCALE);
		uint8_t *p = test.data;

		//to allocate new space   **important
		testData[j] = new uint8_t[imgLimit];

		for (int k = 0; k < imgLimit; ++k)
			testData[j][k] = p[k];

		test.release();
	}

	//to change the form of data from uint8_t* array to CV_32FC1 Mat
	//for pca and svm, the type of data matrix must be CV_32FC1  **important
	Mat testDataMat(testDataSize, imgLimit, CV_32FC1);
	float *testDataMatPtr = (float*)testDataMat.data;
	for (int i = 0; i < testDataSize; ++i)
	{
		int index = i * imgLimit;
		for (int j = 0; j < imgLimit; ++j)
			testDataMatPtr[index + j] = (float)testData[i][j];

		delete[] testData[i];
	}

	
	//to get the features by pca
	Mat compressedTestDataMat;
	int testRows = testDataMat.rows;

	//to get the new test data
	compressedTestDataMat.create(testRows, components, CV_32FC1);

	for (int i = 0; i < testRows; i++)
	{
		Mat vec = testDataMat.row(i), coeffs = compressedTestDataMat.row(i);
		pca.project(vec, coeffs);
	}

	testDataMat.release();
	
	//to write to a file
	fstream fp;
	fp.open("pca.txt", ios::out);
	if (!fp)
	{
		cout << "Fail to open file" << endl;
		return 0;
	}

	float *compressedTrainingDataMatPtr = (float*)compressedTrainingDataMat.data, *compressedTestDataMatPtr = (float*)compressedTestDataMat.data;

	for (int i = 0; i < trainingDataSize; ++i)
	{
		int index = i * components;
		for (int j = 0; j < components; ++j)
		{
			fp << compressedTrainingDataMatPtr[index + j] << endl;
		}
	}

	for (int i = 0; i < testDataSize; ++i)
	{
		int index = i * components;
		for (int j = 0; j < components; ++j)
		{
			fp << compressedTestDataMatPtr[index + j] << endl;
		}
	}

	fp.close();


	//to change the form of labels from int array to CV_32S Mat
	//for svm, the type of labels matrix must be CV_32S  **important (?)
	Mat trainingLabelsMat(trainingDataSize, 1, CV_32S), testLabelsMat(testDataSize, 1, CV_32S);
	int *trainingLabelsMatPtr = (int*)trainingLabelsMat.data, *testLabelsMatPtr = (int*)testLabelsMat.data;

	for (int i = 0; i < trainingDataSize; ++i)
		trainingLabelsMatPtr[i] = trainingLabels[i];

	for (int i = 0; i < testDataSize; ++i)
		testLabelsMatPtr[i] = testLabels[i];

	delete[] trainingLabels;
	delete[] testLabels;

	
	//to ensure that the values are between 0 and 500  (legal?)
	normalize(compressedTrainingDataMat, compressedTrainingDataMat, 0.0, 500.0, NORM_MINMAX);
	normalize(compressedTestDataMat, compressedTestDataMat, 0.0, 500.0, NORM_MINMAX);
	

	//to set some variable for svm
	int resultHeight = 500, resultWidth = 500, resultChannel = 3, resultLimit = resultWidth * resultChannel;
	
	/*
	//the color array  **bgr
	uint8_t bgColors[26][3] =
	{
		(255, 0, 0), (0, 255, 0), (0, 0, 255),
		(255, 255, 0), (255, 0, 255), (0, 255, 255),
		(170, 0, 0), (0, 170, 0), (0, 0, 170),
		(170, 255, 0), (255, 170, 0), (170, 170, 0),
		(170, 0, 255), (255, 0, 170), (170, 0, 170),
		(0, 170, 255), (0, 255, 170), (0, 170, 170),
		(170, 85, 0), (85, 170, 0), (85, 85, 0),
		(170, 0, 85), (85, 0, 170), (85, 0, 85),
		(0, 170, 85), (0, 85, 170)
	};
	//						  TP   TN  FP  FN
	uint8_t testColors[4] = { 255, 0, 170, 85 };
	*/

	//		prediction groundTrue
	int confusion[26][26]{ 0 };

	for (int i = 0; i < 26; ++i)
	{
		for (int j = 0; j < 26; ++j)
		{
			if (confusion[i][j] != 0)
				cout << i << " " << j << "is not 0" << endl;
		}
	}

	//to train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	cout << "svm training" << endl;
	svm->train(compressedTrainingDataMat, ROW_SAMPLE, trainingLabelsMat);
	//svm->train(trainingDataMat, ROW_SAMPLE, trainingLabelsMat);

	//Mat testMat64(1, components, CV_32FC1);
	//float *p64 = (float*)testMat64.data, *compressedTestDataMatPtr = (float*)compressedTestDataMat.data;
	//float *compressedTestDataMatPtr = (float*)compressedTestDataMat.data;

	/*
	// Show the decision regions given by the SVM
	Mat result = Mat::zeros(resultHeight, resultWidth, CV_8UC3);
	uint8_t *pResult = result.data;

	//to draw the basic result
	for (int i = 0; i < resultHeight; ++i)
	{
		int index = i * resultLimit;
		for (int j = 0; j < resultWidth; ++j)
		{
			int k = index + j * resultChannel;
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = svm->predict(sampleMat);

			pResult[k] = bgColors[(int)response][0];
			pResult[k + 1] = bgColors[(int)response][1];
			pResult[k + 2] = bgColors[(int)response][2];

			sampleMat.release();
		}
	}
	*/

	//to show the test data and measure tp tn fp and pn
	//int thickness = -1, lineType = 8, tp = 0, tn = 0, fp = 0, fn = 0;

	//to see the prediction
	for (int i = 0; i < testDataSize; ++i)
	{
		int j = i * components;

		//the type mush be float
		Mat sampleMat(1, components, CV_32FC1);
		//Mat sampleMat = (Mat_<float>(1, 2) << testDataMatPtr[j], testDataMatPtr[j + 1]);
		float *sampleMatPtr = (float*)sampleMat.data;

		for (int k = 0; k < components; ++k)
		{
			sampleMatPtr[k] = compressedTestDataMatPtr[j + k];
		}

		float response = svm->predict(sampleMat);
		//Scalar s;

		//to count the result
		++confusion[(int)response][testLabelsMatPtr[i]];

		/*
		if ((int)response == testLabelsMatPtr[i])
		{
			++tp;
			s = Scalar(testColors[0], testColors[0], testColors[0]);
		}
		else
		{
			++fn;
			s = Scalar(testColors[3], testColors[3], testColors[3]);
		}
		

		circle(result, Point(compressedTestDataMatPtr[j], compressedTestDataMatPtr[j + 1]), 1, s, thickness, lineType);
		*/

		sampleMat.release();
	}

	for (int i = 0; i < 26; ++i)
		cout << 'a' + i << " " << endl;

	//to show the confusion data
	for (int i = 0; i < 26; ++i)
	{
		for (int j = 0; j < 26; ++j)
		{
			cout << confusion[i][j] << " ";
		}
		cout << endl;
	}

	/*
	// Show support vectors
	thickness = 2;
	lineType = 8;
	Mat sv = svm->getUncompressedSupportVectors();
	int svRows = sv.rows;
	for (int i = 0; i < svRows; ++i)
	{
		const float* v = sv.ptr<float>(i);
		circle(result, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
	}

	char s[50];
	sprintf_s(s, 50, "result_type%d_kernel%d_maxIter%d.png", SVM::C_SVC, SVM::LINEAR, 100);
	String resultName(s);

	cout << "test data size = " << testDataSize << " tp = " << tp << " fn = " << fn << endl;
	imwrite(resultName, result);        // save the image
	imshow("SVM Simple Example", result); // show it to the user
	waitKey(0);
	*/

	while (true)
	{
		int type, kernel, maxIter;
		//C_SVC == 100, NU_SVC == 101, ONE_CLASS = 102, EPS_SVR = 103, NU_SVR = 104
		cout << "Please input the SVM type (0 => end)" << endl;
		cin >> type;
		if (type == 0)
			break;

		//CUSTON = -1, LINEAR = 0, POLY = 1, RBF = 2, SIGMOID = 3, CHI2 = 4, INTER = 5
		cout << "Please input the kernel" << endl;
		cin >> kernel;

		cout << "Please input the max iter (0 => end)" << endl;
		cin >> maxIter;

		//to train the SVM
		Ptr<SVM> svm = SVM::create();
		svm->setType(type);
		svm->setKernel(kernel);
		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, maxIter, 1e-6));
		cout << "svm training" << endl;
		svm->train(compressedTrainingDataMat, ROW_SAMPLE, trainingLabelsMat);
		//svm->train(trainingDataMat, ROW_SAMPLE, trainingLabelsMat);

		//Mat testMat64(1, components, CV_32FC1);
		//float *p64 = (float*)testMat64.data, *compressedTestDataMatPtr = (float*)compressedTestDataMat.data;
		//float *compressedTestDataMatPtr = (float*)compressedTestDataMat.data;

		/*
		// Show the decision regions given by the SVM
		Mat result = Mat::zeros(resultHeight, resultWidth, CV_8UC3);
		uint8_t *pResult = result.data;
		
		//to draw the basic result
		for (int i = 0; i < resultHeight; ++i)
		{
			int index = i * resultLimit;
			for (int j = 0; j < resultWidth; ++j)
			{
				int k = index + j * resultChannel;
				Mat sampleMat = (Mat_<float>(1, 2) << j, i);
				float response = svm->predict(sampleMat);

				pResult[k] = bgColors[(int)response][0];
				pResult[k + 1] = bgColors[(int)response][1];
				pResult[k + 2] = bgColors[(int)response][2];

				sampleMat.release();
			}
		}

		//to show the test data and measure tp tn fp and pn
		int thickness = -1, lineType = 8, tp = 0, tn = 0, fp = 0, fn = 0;
		*/

		//to see the prediction
		for (int i = 0; i < testDataSize; ++i)
		{
			int j = i * components;

			//the type mush be float
			//Mat sampleMat = (Mat_<float>(1, 2) << compressedTestDataMatPtr[j], compressedTestDataMatPtr[j + 1]);
			//Mat sampleMat = (Mat_<float>(1, 2) << testDataMatPtr[j], testDataMatPtr[j + 1]);
			
			Mat sampleMat(1, components, CV_32FC1);
			//Mat sampleMat = (Mat_<float>(1, 2) << testDataMatPtr[j], testDataMatPtr[j + 1]);
			float *sampleMatPtr = (float*)sampleMat.data;

			for (int k = 0; k < components; ++k)
			{
				sampleMatPtr[k] = compressedTestDataMatPtr[j + k];
			}
			
			float response = svm->predict(sampleMat);
			//Scalar s;

			//to count the result
			++confusion[(int)response][testLabelsMatPtr[i]];

			/*
			if ((int)response == testLabelsMatPtr[i])
			{
				++tp;
				s = Scalar(testColors[0], testColors[0], testColors[0]);
			}
			else
			{
				++fn;
				s = Scalar(testColors[3], testColors[3], testColors[3]);
			}

			circle(result, Point(compressedTestDataMatPtr[j], compressedTestDataMatPtr[j + 1]), 1, s, thickness, lineType);
			*/

			sampleMat.release();
		}

		for (int i = 0; i < 26; ++i)
			cout << 'a' + i << " " << endl;

		//to show the confusion data
		for (int i = 0; i < 26; ++i)
		{
			for (int j = 0; j < 26; ++j)
			{
				cout << confusion[i][j] << " ";
			}
			cout << endl;
		}

		/*
		// Show support vectors
		thickness = 2;
		lineType = 8;
		Mat sv = svm->getUncompressedSupportVectors();
		int svRows = sv.rows;
		for (int i = 0; i < svRows; ++i)
		{
			const float* v = sv.ptr<float>(i);
			circle(result, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
		}

		char s[50];
		sprintf_s(s, 50, "result_type%d_kernel%d_maxIter%d.png", type, kernel, maxIter);
		String resultName(s);

		cout << "test data size = " << testDataSize << " tp = " << tp << " fn = " << fn << endl;
		imwrite(resultName, result);        // save the image
		imshow("SVM Simple Example", result); // show it to the user
		waitKey(0);
		*/
	}

	cout << "Good Bye" << endl;
	return 0;
}

void GetFiles(string dir, string *files, int fileSize)
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
PCA PCACompression(const Mat& pcaset, int maxComponents, Mat& compressedPcaset)
{
	//pca(input data, Mat() means that PCA engine will compute the mean vector, thr form of eigenvector in matrix, the count of componet)
	PCA pca(pcaset, Mat(), PCA::DATA_AS_ROW, maxComponents);

	// if there is no test data, just return the computed basis, ready-to-use
	//if (!testset.data)
	//	return pca;

	//to ensure if the cols of testset are equal to the cols of pcaset;
	//CV_Assert(testset.cols == pcaset.cols);

	compressedPcaset.create(pcaset.rows, maxComponents, CV_32FC1);
	//compressedTestset.create(testset.rows, maxComponents, CV_32FC1);
	//reconstructedTestset.create(testset.rows, testset.cols, CV_64FC1);

	//to get the new training data
	int rows = pcaset.rows;
	for (int i = 0; i < rows; ++i)
	{
		Mat vec = pcaset.row(i), coeffs = compressedPcaset.row(i);
		pca.project(vec, coeffs);
	}

	/*
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
	*/
	return pca;
}