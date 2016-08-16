#include <iostream>
#include <stdio.h>      /* printf */
#include <math.h>       /* sqrt */
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class ImageProcessor {
public:
    Mat frame;
    vector<Vec4i> hierarchy;
    vector<Point> biggest_blob;
    int workingImgSize = 300;
    Mat workingImg;
    bool useCamera = false;
    bool foundBiggestRect;
    bool foundGrid;
    bool foundCells;

    void resetCheckpoints() {
        foundBiggestRect = false;
        foundGrid = false;
        foundCells = false;
    }

    void displayImage() {
        imshow("Frame", frame);
        imshow("Workingimg", workingImg);
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

//        Mat mask = vertical + horizontal;
        Mat andMask;
        bitwise_and(vertical, horizontal, andMask);
        vector<vector<Point>> joints;
        blur(andMask, andMask, Size(2, 2));
        findContours(andMask, joints, noArray(), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        if (joints.size() >= 90 && joints.size() <= 110) {
            foundGrid = true;
        }
//        printf("There are %lu joints\n", joints.size());

//        // Refine mask
//        Mat kernel = Mat::ones(2, 2, CV_8UC1);
//        dilate(mask, mask, kernel);
//        blur(mask, mask, Size(2,2));
//        Mat masked = ~(workingImg - mask);
//        dilate(masked, masked, Mat());
//        return masked;
    }

    void findCells() {
        vector<vector<Point>> allContours;
        vector<vector<Point>> cellContours;
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
                    printf("Size: %f\n", area);
                    cellContours.push_back(allContours[c]);
                    //TODO: remove temp draw below
                    drawRect(allContours, c, Scalar(255, 255, 0));
                }
            }
        }
        printf("There are %lu cells\n", cellContours.size());
        if (cellContours.size() == 81) foundCells = true;
    }

private:

    void printPoints(vector<Point> points) {
        printf(">>>%lu\n\n", points.size());
        for (int i=0; i < points.size(); i++) {
            printf("%d: %d, %d\n", i, points[i].x, points[i].y);
        }
    }

    void drawBiggestRect(vector<vector<Point>> contours, int index) {
        drawContours(frame, contours, index, Scalar( 255, 255, 255 ), 2, 8, hierarchy, 0, Point());
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
//        printf("max width: %i\nmax height: %i\n", max_width, max_height);

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
};