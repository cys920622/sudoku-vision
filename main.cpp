#include <iostream>
#include "opencv2/opencv.hpp"
#include "image_processing.cpp"

using namespace cv;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    ImageProcessor imageProcessor;
    for(;;)
    {
        imageProcessor.capture_and_process(cap);
//        imshow("Frame", imageProcessor.frame);
//        imshow("Out", imageProcessor.out);

        // Quit on key press
        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}