#include <iostream>
#include "opencv2/opencv.hpp"
#include "imageProcessor.cpp"
#include "sudokuSolver.cpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    printf("START\n");

    VideoCapture cap(0);
    ImageProcessor ip;
    Solver solver;
    bool isTraining = false;

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            string arg = argv[i];
            if (arg == "train") {
                printf("Creating sample data\n");
                isTraining = true;
            } else if (arg == "sample") {
                printf("Using sample static image\n");
                ip.useCamera = false;
            } else {
                cerr << arg << " is not a valid option.\n";
                return -1;
            }
        }
    }

    for (int i = 0;; i++) {
        while (true) {
            ip.resetCheckpoints();
            ip.findBiggestRect(cap);
            if (!ip.foundBiggestRect) {
                ip.displayImage();
                break;
            }
            ip.findGrid();
            if (!ip.foundGrid) {
                ip.displayImage();
                break;
            }
            ip.findCells();
            if (!ip.foundCells) {
                ip.displayImage();
                break;
            }
            if (isTraining) {
                ip.createSampleClassifications();
                printf("Training completed successfully.\n");
                return 0;
            }

            ip.trainKnn();
            ip.readGrid();
            if (!ip.puzzleReadComplete) {
                ip.displayImage();
                break;
            }

            ip.displaySolution(solver.solveSudoku(ip.grid));
            break;
        }
        // Quit on pressing ESC
        if (waitKey(30) == 27) break;
    }

    return 0;
}