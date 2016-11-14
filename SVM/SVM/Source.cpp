#include <iostream>
#include <vector>
#include <dirent.h>
#include <string>
#include <fstream>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <cmath>

#include <opencv2/opencv.hpp>

#define trainingFileSize 7048
#define testFileSize 1302
#define imgHeight 227
#define imgWidth 227
#define imgLimit imgHeight * imgWidth

using namespace cv;
using namespace cv::ml;
using namespace std;

void GetFiles(string dir, string *files, int fileSize); 

int main()
{
	//to get the training data
	string dir = string("./CSL/training");
	string *files = new string[trainingFileSize];
	GetFiles(dir, files, trainingFileSize);

	//the type of labels must be int (CV_32S)  (?)
	int *trainingLabels = new int [trainingFileSize - 2];
	uint8_t *trainingData[trainingFileSize - 2];
	int trainingDataSize = trainingFileSize - 2;

	//to get training data & labels
	//to skip . & ..
	for (int i = 2; i < trainingFileSize; ++i)
	{
		int j = i - 2;

		trainingLabels[j] = files[i][0] - 97;

		//to use grayscale
		Mat training = imread(dir + "/" + files[i], CV_LOAD_IMAGE_GRAYSCALE);
		uint8_t *p = training.data;

		//to allocate new space   **important
		trainingData[j] = new uint8_t[imgLimit];

		for (int k = 0; k < imgLimit; ++k)
			trainingData[j][k] = p[k];

		training.release();
	}

	//to change the form of data from uint8_t* array to CV_32FC1 Mat
	//for svm, the type of data matrix must be CV_32FC1  **important
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

	//to get test data
	dir = string("./CSL/test");
	
	delete[] files;
	files = new string[testFileSize];
	GetFiles(dir, files, testFileSize);

	//the type of labels must be int (CV_32S)
	int *testLabels = new int[testFileSize - 2];
	uint8_t *testData[testFileSize - 2];
	int testDataSize = testFileSize - 2;

	//to get test data & labels
	for (int i = 2; i < testFileSize; ++i)
	{
		int j = i - 2;

		testLabels[j] = files[i][0] - 97;

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
	//for svm, the type of data matrix must be CV_32FC1  **important
	Mat testDataMat(testDataSize, imgLimit, CV_32FC1);
	float *testDataMatPtr = (float*)testDataMat.data;
	for (int i = 0; i < testDataSize; ++i)
	{
		int index = i * imgLimit;
		for (int j = 0; j < imgLimit; ++j)
			testDataMatPtr[index + j] = (float)testData[i][j];

		delete[] testData[i];
	}

	//to change the form of labels from int array to CV_32S Mat
	//for svm, the type of labels matrix must be CV_32S  **important
	Mat trainingLabelsMat(trainingDataSize, 1, CV_32S), testLabelsMat(testDataSize, 1, CV_32S);
	int *trainingLabelsMatPtr = (int*)trainingLabelsMat.data, *testLabelsMatPtr = (int*)testLabelsMat.data;

	for (int i = 0; i < trainingDataSize; ++i)
		trainingLabelsMatPtr[i] = trainingLabels[i];

	for (int i = 0; i < testDataSize; ++i)
		testLabelsMatPtr[i] = testLabels[i];

	delete[] trainingLabels;
	delete[] testLabels;

	double fromScalingValue = 0.0, toScalingValue = 1.0;
	//to ensure that the values are between 0 and 500  **scaling
	normalize(trainingDataMat, trainingDataMat, fromScalingValue, toScalingValue, NORM_MINMAX);
	normalize(testDataMat, testDataMat, fromScalingValue, toScalingValue, NORM_MINMAX);
	
	//		prediction groundTrue
	int confusion[26][26]{ 0 };
	float precision[3][26], recall[3][26];
	fstream fp;
	char fileName[100];

	while (true)
	{
		int type, kernel, maxIter, gamma, c;

		//C_SVC == 100, NU_SVC == 101, ONE_CLASS = 102, EPS_SVR = 103, NU_SVR = 104
		cout << "Please input the SVM type (0 => end)" << endl;
		cin >> type;
		if (type == 0)
			break;

		//CUSTON = -1, LINEAR = 0, POLY = 1, RBF = 2, SIGMOID = 3, CHI2 = 4, INTER = 5
		cout << "Please input the kernel" << endl;
		cin >> kernel;

		//1 by default
		//cout << "Please input the gamma" << endl;
		//cin >> gamma;

		//0 by default
		//cout << "Please input the c" << endl;
		//cin >> c;

		cout << "Please input the max iter" << endl;
		cin >> maxIter;

		//to train the SVM
		Ptr<SVM> svm = SVM::create();
		svm->setType(type);
		svm->setKernel(kernel);
		//svm->setGamma(0.001);
		//svm->setC(0.01);
		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, maxIter, 1e-6));
		cout << "svm training" << endl;

		//to set timer
		clock_t start, finish;
		start = clock();
		svm->train(trainingDataMat, ROW_SAMPLE, trainingLabelsMat);

		//to initialize the confusion array
		for (int i = 0; i < 26; ++i)
			for (int j = 0; j < 26; ++j)
				confusion[i][j] = 0;

		//to see the prediction
		for (int i = 0; i < testDataSize; ++i)
		{
			int j = i * imgLimit;

			Mat sampleMat(1, imgLimit, CV_32FC1);
			float *sampleMatPtr = (float*)sampleMat.data;

			for (int k = 0; k < imgLimit; ++k)
				sampleMatPtr[k] = testDataMatPtr[j + k];
			
			float response = svm->predict(sampleMat);

			//to count the result
			++confusion[(int)response][testLabelsMatPtr[i]];

			sampleMat.release();
		}

		finish = clock();

		//to calculate the training time
		cout << "training time = " << (double)(finish - start) / CLOCKS_PER_SEC << endl;

		for (int i = 0; i < 26; ++i)
			cout << (char)('a' + i) << " ";
		cout << endl;

		sprintf_s(fileName, 100, "confusion_type%d_kernel%d_maxIter%d.txt", type, kernel, maxIter);
		fp.open(fileName, ios::out);
		if (!fp)
		{
			cerr << "File cannot open" << endl;
			return 0;
		}

		//to show the confusion data
		for (int i = 0; i < 26; ++i)
		{
			for (int j = 0; j < 26; ++j)
			{
				cout << confusion[i][j] << " ";
				fp << confusion[i][j] << " ";
			}
			cout << endl;
			fp << endl;
		}

		fp.close();

		sprintf_s(fileName, 100, "precision_type%d_kernel%d_maxIter%d.txt", type, kernel, maxIter);
		fp.open(fileName, ios::out);
		if (!fp)
		{
			cerr << "File cannot open" << endl;
			return 0;
		}

		//to calculate each precision
		cout << "precision : " << endl;
		for (int i = 0; i < 26; ++i)
		{
			precision[maxIter - 1][i] = 0.0;
			int sum = 0;
			for (int j = 0; j < 26; ++j)
				sum += confusion[i][j];

			//to ensure that the precision won't be too small 
			if (sum >= 1 && confusion[i][i] != 0)
				precision[maxIter - 1][i] = (float)confusion[i][i] / sum;

			cout << precision[maxIter - 1][i] << " ";
			fp << precision[maxIter - 1][i] << endl;

			if (i == 12)
				cout << endl;
		}

		cout << endl;
		fp.close();

		sprintf_s(fileName, 100, "recall_type%d_kernel%d_maxIter%d.txt", type, kernel, maxIter);
		fp.open(fileName, ios::out);
		if (!fp)
		{
			cerr << "File cannot open" << endl;
			return 0;
		}

		//to calculate the recall
		cout << "recall : " << endl;
		for (int i = 0; i < 26; ++i)
		{
			recall[maxIter - 1][i] = 0.0;
			int sum = 0;
			for (int j = 0; j < 26; ++j)
				sum += confusion[j][i];

			//to ensure that the recall won't be too small 
			if (sum >= 1 && confusion[i][i] != 0)
				recall[maxIter - 1][i] = (float)confusion[i][i] / sum;

			cout << recall[maxIter - 1][i] << " ";
			fp << recall[maxIter - 1][i] << endl;

			if (i == 12)
			{
				cout << endl;
			}
		}

		cout << endl;
		fp.close();
	}

	sprintf_s(fileName, 100, "averagePrecision_type%d_kernel%d.txt", 100, 0);
	fp.open(fileName, ios::out);
	if (!fp)
	{
		cerr << "File cannot open" << endl;
		return 0;
	}

	cout << "average precision" << endl;
	for (int i = 0; i < 26; ++i)
	{
		float sum = 0.0;
		for (int j = 0; j < 3; ++j)
			sum += precision[j][i];

		cout << sum / 3 << " ";
		fp << sum / 3 << endl;

		if (i == 12)
			cout << endl;
	}
	
	cout << endl;
	fp.close();

	sprintf_s(fileName, 100, "averageRecall_type%d_kernel%d.txt", 100, 0);
	fp.open(fileName, ios::out);
	if (!fp)
	{
		cerr << "File cannot open" << endl;
		return 0;
	}

	cout << "average recall" << endl;
	for (int i = 0; i < 26; ++i)
	{
		float sum = 0.0;
		for (int j = 0; j < 3; ++j)
			sum += recall[j][i];

		cout << sum / 3 << " ";
		fp << sum / 3 << endl;

		if (i == 12)
			cout << endl;
	}

	cout << endl;
	fp.close();

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
	for (int i = 0; i < fileSize; ++i)
		if ((dirp = readdir(dp)) != NULL)
			files[i] = string(dirp->d_name);

	//to close directory pointer
	closedir(dp);
	return;
}