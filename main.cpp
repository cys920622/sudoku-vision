#include <iostream>
#include "opencv2/opencv.hpp"
#include "imageProcessor.cpp"
#include "sudokuSolver.cpp"

using namespace cv;
using namespace std;

int main(int, char**) {
    printf("START\n");

    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    ImageProcessor ip;
    Solver solver;
    bool isTraining = false;

    for (int i = 0;; i++) {
//    for (int i = 0; i < 1; i++) { // One iteration
        while (true) {
            ip.resetCheckpoints();
            ip.findBiggestRect(cap);
            if (!ip.foundBiggestRect) {
//                printf("1: finding rect: %d\n", i);
                ip.displayImage();
                break;
            }
            ip.findGrid();
            if (!ip.foundGrid) {
//                printf("2: finding grid: %d\n", i);
                ip.displayImage();
                break;
            }
            ip.findCells();
            if (!ip.foundCells) {
//                printf("3: finding cells: %d\n", i);
                ip.displayImage();
                break;
            }
            if (isTraining) {
                ip.createSampleClassifications();
                printf("Training completed successfully.\n");
                break;
            }

            ip.trainKnn();
            ip.readGrid();
            if (!ip.puzzleReadComplete) {
//                printf("4: reading grid: %d\n", i);
                ip.displayImage();
                break;
            }

            ip.displaySolution(solver.solveSudoku(ip.grid));
            break;
        }
        // Quit on key press
        if (waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}