#include <iostream>
#include "opencv2/opencv.hpp"
#include "imageProcessor.cpp"

using namespace cv;
using namespace std;

int main(int, char**) {
    printf("START");
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    ImageProcessor imageProcessor;

    for (int i = 0;; i++) {
        while (true) {
            imageProcessor.resetCheckpoints();
            imageProcessor.findBiggestRect(cap);
            if (!imageProcessor.foundBiggestRect) {
                printf("FAIL: find rect: %d\n", i);
                break;
            }
            imageProcessor.findGrid();
            if (!imageProcessor.foundGrid) {
                printf("FAIL: find grid: %d\n", i);
                break;
            }
            imageProcessor.findCells();
            if (!imageProcessor.foundCells) {
                printf("FAIL: find cells: %d\n", i);
                break;
            }

            imageProcessor.displayImage();
            break;
        }
        // Quit on key press
        if (waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}