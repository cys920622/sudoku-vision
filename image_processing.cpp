#include <iostream>
#include <stdio.h>      /* printf */
#include <math.h>       /* sqrt */
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class ImageProcessor {
public:
    Mat frame;
    Mat gray;
    Mat canny;
    Mat th_gaussian;
    Mat contoursImg;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<Point> biggest_blob;
    Mat transformed;
    Mat out;
    bool isVideoOn = false;

    void capture_and_process( VideoCapture video ) {
        int thresh = 100;
        //TODO: remove mock image
        if (isVideoOn) {
            video >> frame;
        } else {
            frame = imread("sudoku_webcam.jpeg",1);
        }
        cvtColor(frame, gray, CV_BGR2GRAY);
        //TODO: remove magic numbers
        GaussianBlur(gray, gray, Size(7,7), 1.5, 1.5);
        adaptiveThreshold(~gray, th_gaussian, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
            THRESH_BINARY, 15, -2);

        Canny(th_gaussian, canny, thresh, thresh*2, 3 );
        contoursImg = canny.clone();
        findContours(contoursImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        find_biggest_blob();

        out = remove_lines(out.clone());
    }

    // http://answers.opencv.org/question/63847/how-to-extract-tables-from-an-image/
    Mat remove_lines(Mat img) {
        Mat horizontal = img.clone();
        Mat vertical = img.clone();
        int scale = 10;
        int horizontal_size = horizontal.cols / scale;
        Mat horizontal_structure = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));

        erode(horizontal, horizontal, horizontal_structure, Point(-1, -1));
        dilate(horizontal, horizontal, horizontal_structure, Point(-1, -1), 3);


        int vertical_size = vertical.cols / scale;
        Mat vertical_structure = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
        erode(vertical, vertical, vertical_structure, Point(-1, -1));
        dilate(vertical, vertical, vertical_structure, Point(-1, -1), 3);

        Mat mask = vertical + horizontal;
        // -------------------SANDBOX BEGIN--------------------
        Mat andMask;
        bitwise_and(vertical, horizontal, andMask);
        vector<vector<Point>> joints;
        blur(andMask, andMask, Size(2, 2));
        findContours(andMask, joints, noArray(), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        imshow("Joints", andMask);
        printf("There are %lu joints\n", joints.size());
        // -------------------SANDBOX END----------------------
        // Refine mask
        Mat kernel = Mat::ones(2, 2, CV_8UC1);
        dilate(mask, mask, kernel);
//        blur(mask, mask, Size(2,2));

        Mat masked = ~(img - mask);

//        dilate(masked, masked, Mat());
        return masked;
    }
    void find_biggest_blob() {
        double max_area = 0;
        for (int c = 0; c < contours.size(); c++)
        {
            double area = contourArea(contours[c]);
            if (area > 10000)
            {
                double perimeter = arcLength(contours[c], true);
                vector<Point> approx;
                approxPolyDP(contours[c], approx, 0.02*perimeter, true);
                if (area > max_area && approx.size() == 4)
                {
                    max_area = area;
                    biggest_blob = order_points(approx);
                    out = transform_and_resize(th_gaussian, biggest_blob);
                    Scalar color = Scalar( 255, 255, 255 );
                    drawContours(frame, contours, c, color, 2, 8, hierarchy, 0, Point());
                }
            }
        }
    }

    Mat transform_and_resize(Mat image, vector<Point> rect) {
        transformed = four_point_transform(image, rect);
        Mat out;
        resize(transformed, out, Size(1000, 1000), 0, 0, CV_INTER_CUBIC);
        return out;
    }

    vector<Point> order_points(vector<Point> points) {
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

    Mat four_point_transform(Mat image, vector<Point> ordered_points) {
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

    void print_points(vector<Point> points) {
        printf(">>>%lu\n\n", points.size());
        for (int i=0; i < points.size(); i++) {
            printf("%d: %d, %d\n", i, points[i].x, points[i].y);
        }
    }
};