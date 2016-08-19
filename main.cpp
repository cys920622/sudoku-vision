#include <iostream>
#include "opencv2/opencv.hpp"
#include "imageProcessor.cpp"

using namespace cv;
using namespace std;

int main(int, char**) {
    printf("START\n");

    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    ImageProcessor imageProcessor;
    bool isTraining = false;

    for (int i = 0;; i++) {
//    for (int i = 0; i < 1; i++) { // One iteration
        while (true) {
            imageProcessor.resetCheckpoints();
            imageProcessor.findBiggestRect(cap);
            if (!imageProcessor.foundBiggestRect) {
//                printf("1: finding rect: %d\n", i);
                imageProcessor.displayImage();
                break;
            }
            imageProcessor.findGrid();
            if (!imageProcessor.foundGrid) {
//                printf("2: finding grid: %d\n", i);
                imageProcessor.displayImage();
                break;
            }
            imageProcessor.findCells();
            if (!imageProcessor.foundCells) {
//                printf("3: finding cells: %d\n", i);
                imageProcessor.displayImage();
                break;
            }
            if (isTraining) {
                imageProcessor.createSampleClassifications();
                printf("Training completed successfully.\n");
                break;
            }

            imageProcessor.trainKnn();
            imageProcessor.readGrid();
            if (!imageProcessor.puzzleReadComplete) {
                printf("4: reading grid: %d\n", i);
                imageProcessor.displayImage();
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