#include <iostream>
#include <stdio.h>      /* printf */
#include <math.h>       /* sqrt */
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;

class ImageProcessor {
public:
    Mat frame;
    vector<Vec4i> hierarchy;
    vector<Point> biggest_blob;
    int workingImgSize = 300;
    Mat workingImg;
    bool useCamera = true;
    bool foundBiggestRect;
    bool foundGrid;
    bool foundCells;
    bool puzzleReadComplete;
    vector<vector<Point>> cellContours;
    vector<Mat> cells;
    int resizedCellWidth = 20;
    int resizedCellHeight = 30;
    Ptr<KNearest> kNearest;
    vector<int> grid;

    void resetCheckpoints() {
        foundBiggestRect = false;
        foundGrid = false;
        foundCells = false;
        puzzleReadComplete = false;
    }

    void displayImage() {
        imshow("Frame", frame);
//        imshow("Workingimg", workingImg);
    }

    void findBiggestRect(VideoCapture video) {
        vector<vector<Point> > contours;
        int thresh = 100;
        if (useCamera) {
            video >> frame;
        } else {
            frame = imread("sudoku_sample.png",1);
        }
        cvtColor(frame, workingImg, CV_BGR2GRAY);
        GaussianBlur(workingImg, workingImg, Size(7,7), 1.5, 1.5);
        adaptiveThreshold(~workingImg, workingImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
            THRESH_BINARY, 15, -2);
        Mat canny;
        Canny(workingImg, canny, thresh, thresh*2, 3 );
        findContours(canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        double max_area = 0;
        int contourIndex = 0;
        for (int c = 0; c < contours.size(); c++) {
            double area = contourArea(contours[c]);
            if (area > 10000) {
                double perimeter = arcLength(contours[c], true);
                vector<Point> approx;
                approxPolyDP(contours[c], approx, 0.02*perimeter, true);
                if (area > max_area && approx.size() == 4) {
                    max_area = area;
                    biggest_blob = orderPoints(approx);
                    workingImg = transformAndResize(workingImg.clone(), biggest_blob);
                    contourIndex = c;
                    foundBiggestRect = true;
                }
            }
        }
        if (foundBiggestRect) drawBiggestRect(contours, contourIndex);
    }

    // http://answers.opencv.org/question/63847/how-to-extract-tables-from-an-image/
    void findGrid() {
        Mat horizontal = workingImg.clone();
        Mat vertical = workingImg.clone();
        int scale = 2;
        int horizontal_size = horizontal.cols / scale;
        Mat horizontal_structure = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));

        erode(horizontal, horizontal, horizontal_structure, Point(-1, -1));
        dilate(horizontal, horizontal, horizontal_structure, Point(-1, -1), 3);

        int vertical_size = vertical.cols / scale;
        Mat vertical_structure = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
        erode(vertical, vertical, vertical_structure, Point(-1, -1));
        dilate(vertical, vertical, vertical_structure, Point(-1, -1), 3);

        Mat andMask;
        bitwise_and(vertical, horizontal, andMask);
        vector<vector<Point>> joints;
        blur(andMask, andMask, Size(2, 2));
        findContours(andMask, joints, noArray(), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        if (joints.size() >= 90 && joints.size() <= 110) {
            foundGrid = true;
        }
    }

    void findCells() {
        vector<vector<Point>> allContours;
        vector<vector<Point>> localCellContours;
        int thresh = 100;
        int areaThreshold = 800;
        findContours(workingImg.clone(), allContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        for (int c = 0; c < allContours.size(); c++) {
            double area = contourArea(allContours[c]);
            if (area > areaThreshold && area < areaThreshold*1.3) {
                double perimeter = arcLength(allContours[c], true);
                vector<Point> approx;
                approxPolyDP(allContours[c], approx, 0.02*perimeter, true);
                if (approx.size() == 4) {
                    localCellContours.push_back(orderPoints(approx));
                }
            }
        }
        if (localCellContours.size() == 81) {
            cellContours = orderCellContours(localCellContours);
            foundCells = true;
        }
    }

    void createSampleClassifications() {
        Mat responseInts;
        Mat flattenedFloatImages;

        Mat workingImgColor;
        cvtColor(workingImg.clone(), workingImgColor, cv::COLOR_GRAY2BGR);
        int cropPixels = 2;
        int cellAreaThreshold = 40;
        for (int i = 0; i < cellContours.size(); i++) {
            Rect roiRect = cropContourToRect(cellContours[i], cropPixels);
            vector<vector<Point>> numberContours;
            Mat imgRoi(workingImg, roiRect);

            findContours(imgRoi.clone(), numberContours, noArray(), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

            if (numberContours.size() == 0 || contourArea(numberContours[0]) < cellAreaThreshold) continue;

            rectangle(workingImgColor, roiRect, Scalar(0, 0, 255), 2);

            Mat imgRoiResized;
            resize(imgRoi, imgRoiResized, Size(resizedCellWidth, resizedCellHeight));

            imshow("workingImg", workingImgColor);
            imshow("imgroi", imgRoi); //TODO: remove temp
            int c = waitKey(0);
            if (c == 27) {
                return;
            } else {
                responseInts.push_back(c - 48);
                Mat matImageFloat;
                imgRoiResized.convertTo(matImageFloat, CV_32FC1);
                Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);
                flattenedFloatImages.push_back(matImageFlattenedFloat);
            }

            rectangle(workingImgColor, roiRect, Scalar(0, 255, 0), 2);
        }

        FileStorage classificationsFs("classifications.xml", FileStorage::WRITE);
        if (!classificationsFs.isOpened()) {
            cout << "Error opening classification file!\n\n";
            return;
        }
        classificationsFs << "classifications" << responseInts;
        classificationsFs.release();

        FileStorage trainingImagesFs("images.xml", FileStorage::WRITE);
        if (!trainingImagesFs.isOpened()) {
            cout << "Error opening training images file!\n\n";
            return;
        }
        trainingImagesFs << "images" << flattenedFloatImages;
        trainingImagesFs.release();
    }

    void trainKnn() {
        Mat classificationInts;
        Mat flattenedFloatImages;
        FileStorage classificationsFs("classifications.xml", FileStorage::READ);
        if (!classificationsFs.isOpened()) {
            cout << "Error opening classification file!\n\n";
            return;
        }
        classificationsFs["classifications"] >> classificationInts;
        classificationsFs.release();

        FileStorage trainingImagesFs("images.xml", FileStorage::READ);
        if (!trainingImagesFs.isOpened()) {
            cout << "Error opening training images file!\n\n";
            return;
        }
        trainingImagesFs["images"] >> flattenedFloatImages;
        trainingImagesFs.release();


        kNearest = KNearest::create();
        kNearest->train(flattenedFloatImages, ROW_SAMPLE, classificationInts);
    }

    void readGrid() {
        vector<int> sudokuGrid;
        Mat workingImgColor;
        cvtColor(workingImg.clone(), workingImgColor, cv::COLOR_GRAY2BGR);
        int cropPixels = 2;
        int cellAreaThreshold = 40;
        for (int i = 0; i < cellContours.size(); i++) {
            Rect roiRect = cropContourToRect(cellContours[i], cropPixels);
            vector<vector<Point>> numberContours;
            Mat imgRoi(workingImg, roiRect);

            findContours(imgRoi.clone(), numberContours, noArray(), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

            if (numberContours.size() == 0 || contourArea(numberContours[0]) < cellAreaThreshold) {
                sudokuGrid.push_back(0);
            } else {
                sudokuGrid.push_back(readCell(imgRoi));
            };
        }
        grid = sudokuGrid;
        puzzleReadComplete = true;
//        printGrid(grid);
    }

private:

    Rect cropContourToRect(vector<Point> contour, int cropPixels) {
        Rect cropped(
                contour[0].x + cropPixels,
                contour[0].y + cropPixels,
                contour[2].x - contour[0].x - cropPixels*2,
                contour[2].y - contour[0].y - cropPixels*2
        );
        return cropped;
    }

    void printPoints(vector<Point> points) {
        printf(">>>%lu\n\n", points.size());
        for (int i=0; i < points.size(); i++) {
            printf("%d: %d, %d\n", i, points[i].x, points[i].y);
        }
    }

    void drawBiggestRect(vector<vector<Point>> contours, int index) {
        Scalar color;
        if (foundCells) {
            color = Scalar(0, 255, 0);
        } else {
            color = Scalar(0, 0, 255);
        }
        drawContours(frame, contours, index, color, 2, 8, hierarchy, 0, Point());
    }

    void drawRect(vector<vector<Point>> contours, int index, Scalar color) {
        drawContours(frame, contours, index, color, 1, 8, hierarchy, 0, Point());
    }

    Mat transformAndResize(Mat image, vector<Point> rect) {
        Mat transformed = fourPointTransform(image, rect);
        Mat out;
        resize(transformed, out, Size(workingImgSize, workingImgSize), 0, 0, CV_INTER_CUBIC);
        return out;
    }

    vector<Point> orderPoints(vector<Point> points) {
        vector<Point> ordered(4);

        auto tl_br_result = minmax_element(points.begin(), points.end(),
                                           [](Point tl, Point br) {
                                               return (tl.x + tl.y) < (br.x + br.y);
                                           });
        ordered[0] = points[tl_br_result.first - points.begin()];
        ordered[2] = points[tl_br_result.second - points.begin()];

        auto tr_bl_result = minmax_element(points.begin(), points.end(),
                                           [](Point tr, Point bl) {
                                               return (tr.x - tr.y) > (bl.x - bl.y);
                                           });
        ordered[1] = points[tr_bl_result.first - points.begin()];
        ordered[3] = points[tr_bl_result.second - points.begin()];

        return ordered;
    }

    Mat fourPointTransform(Mat image, vector<Point> ordered_points) {
        // order is tl, tr, br, bl
        Point tl = ordered_points[0];
        Point tr = ordered_points[1];
        Point br = ordered_points[2];
        Point bl = ordered_points[3];

        int max_width = (int) max(
                sqrt(pow((br.x - bl.x), 2) + pow((br.y - bl.y), 2)),
                sqrt(pow((tr.x - tl.x), 2) + pow((tr.y - tl.y), 2))
        );
        int max_height = (int) max(
                sqrt(pow((tr.x - br.x), 2) + pow((tr.y - br.y), 2)),
                sqrt(pow((tl.x - bl.x), 2) + pow((tl.y - bl.y), 2))
        );

        // Construct new points
        Point2f inputQuad[4];
        inputQuad[0] = Point2f(tl.x, tl.y);
        inputQuad[1] = Point2f(tr.x, tr.y);
        inputQuad[2] = Point2f(br.x, br.y);
        inputQuad[3] = Point2f(bl.x, bl.y);
        Point2f outputQuad[4];
        outputQuad[0] = Point2f(0, 0);
        outputQuad[1] = Point2f(max_width, 0);
        outputQuad[2] = Point2f(max_width, max_height);
        outputQuad[3] = Point2f(0, max_height);

        Mat matrix = getPerspectiveTransform(inputQuad, outputQuad);
        Mat warped;
        warpPerspective(image, warped, matrix, Size(max_width, max_height));
        // Compute and apply transformation
        return warped;
    }

    /**
     * Read the value of a single sudoku cell
     * @param cell - processed grayscale image of a cell
     */
    int readCell(Mat cell) {
        resize(cell, cell, Size(resizedCellWidth, resizedCellHeight));
        Mat matFloat;
        cell.convertTo(matFloat, CV_32FC1);
        Mat matFlattenedFloat = matFloat.reshape(1, 1);
        Mat matCurrentDigit(0, 0, CV_32F);
        kNearest->findNearest(matFlattenedFloat, 1, matCurrentDigit);
        float fltCurrentDigit = (float)matCurrentDigit.at<float>(0, 0);
        int finalDigit = int(fltCurrentDigit);
        return finalDigit;
    }

    /**
     * Order cell contours lexocigraphically
     * @param contours
     * @return
     */
    vector<vector<Point>> orderCellContours(vector<vector<Point>> contours) {
        vector<vector<Point>> sortedContours(contours);
        sort(sortedContours.begin(), sortedContours.end(), [](vector<Point> a, vector<Point> b) -> bool {
            return a[0].y < b[0].y;
        });
        for (int i=0; i < contours.size(); i+=9) {
            sort(sortedContours.begin()+i, sortedContours.begin()+i+9, [](vector<Point> a, vector<Point> b) -> bool {
                return a[0].x < b[0].x;
            });
        }
        return sortedContours;
    }

    void printGrid(vector<int> grid) {
        printf("\n----------------------------------\n");
        for (int i=0; i < grid.size(); i++) {
            printf("%i", grid[i]);
            if ((i+1)%9 == 0) {
                printf("\n----------------------------------\n");
            } else {
                printf(" | ");
            }
        }
    }
};