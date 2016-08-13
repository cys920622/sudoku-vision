#include <iostream>
#include <stdio.h>      /* printf */
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main() {
    Mat thr,gray,con;
    Mat src=imread("train2me.png",1);
    cvtColor(src,gray,CV_BGR2GRAY);
    threshold(gray,thr,200,255,THRESH_BINARY_INV); // Threshold to create input
    thr.copyTo(con);


// Read stored sample and label for training
    Mat sample;
    Mat response,tmp;
    FileStorage Data("TrainingData.yml",FileStorage::READ); // Read traing data to a Mat
    Data["data"] >> sample;
    Data.release();

    FileStorage Label("LabelData.yml",FileStorage::READ); // Read label data to a Mat
    Label["label"] >> response;
    Label.release();


    Ptr<ml::KNearest> knn = ml::KNearest::create();
    knn->train(sample,response); // Train with sample and responses
    cout<<"Training compleated.....!!"<<endl;

    vector< vector <Point> > contours; // Vector for storing contour
    vector< Vec4i > hierarchy;

//Create input sample by contour finding and cropping
    findContours( con, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    Mat dst(src.rows,src.cols,CV_8UC3,Scalar::all(0));

    for( int i = 0; i< contours.size(); i=hierarchy[i][0] ) // iterate through each contour for first hierarchy level .
    {
        Rect r= boundingRect(contours[i]);
        Mat ROI = thr(r);
        Mat tmp1, tmp2;
        resize(ROI,tmp1, Size(10,10), 0,0,INTER_LINEAR );
        tmp1.convertTo(tmp2,CV_32FC1);
        float p=knn->findNearest(tmp2.reshape(1,1), 1, tmp2);
        char name[4];
        sprintf(name,"%d",(int)p);
        putText( dst,name,Point(r.x,r.y+r.height) ,0,1, Scalar(0, 255, 0), 2, 8 );
    }

    imshow("src",src);
    imshow("dst",dst);
    imwrite("dest.jpg",dst);
    waitKey();
    return 0;
}