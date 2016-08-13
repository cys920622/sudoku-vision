#include <iostream>
#include <stdio.h>      /* printf */
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

// http://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python
int main() {
    Mat src, gray, thr, con, sample, response_array;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    src = imread("train2me.png", 1);
    cvtColor(src, gray, CV_BGR2GRAY);
    threshold(gray,thr,200,255,THRESH_BINARY_INV);
    thr.copyTo(con);

    findContours(con, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    for(int i = 0; i < contours.size(); i=hierarchy[i][0]) {
        Rect r = boundingRect(contours[i]);
        rectangle(src,Point(r.x,r.y), Point(r.x+r.width,r.y+r.height), Scalar(0,0,255),2,8,0);
        Mat ROI = thr(r);
        Mat tmp1, tmp2;
        resize(ROI,tmp1, Size(10,10), 0,0,INTER_LINEAR );
        tmp1.convertTo(tmp2,CV_32FC1); //convert to float
        sample.push_back(tmp2.reshape(1,1)); // Store  sample data
        imshow("src",src);
        int c=waitKey(0); // Read corresponding label for contour from keyoard
        c-=0x30;     // Convert ascii to integer value
        response_array.push_back(c); // Store label to a mat
        rectangle(src,Point(r.x,r.y), Point(r.x+r.width,r.y+r.height), Scalar(0,255,0),2,8,0);
    }

    // Store the data to file
    Mat response,tmp;
    tmp=response_array.reshape(1,1); //make continuous
    tmp.convertTo(response,CV_32FC1); // Convert  to float

    FileStorage Data("TrainingData.yml",FileStorage::WRITE); // Store the sample data in a file
    Data << "data" << sample;
    Data.release();

    FileStorage Label("LabelData.yml",FileStorage::WRITE); // Store the label data in a file
    Label << "label" << response;
    Label.release();
    cout<<"Training and Label data created successfully....!! "<<endl;

    return 0;
}