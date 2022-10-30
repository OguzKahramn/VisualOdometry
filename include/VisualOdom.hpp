#pragma once

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <ctype.h>
#include <algorithm> 
#include <iterator> 
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#define MIN_NUM_FEAT 2000

using namespace std;
using namespace cv;

class VisualOdom{
  public:
    Mat img1, img2, img1gry, img2gry, prevImg, currImg, currImgGry;  // images variables
    Mat rotM, transM, E, R, T, mask, traj; //matrix variables 
    char imgName1[100], imgName2[100] , text[100], imgName[100]; // buffer for sprintf
    double scale;
    std::string main_path, sequence_number;   // dataset path and KITTI sequence number
    int max_frame;                           
    vector<Point2f> points1, points2, prevFeatures, currFeatures;        
    vector<uchar> status;
    double focal;  // 1 - 1 
    Point2d pp; // 1 - 3    1 - 7
    
    // functions
    VisualOdom(std::string main_path, std::string sequence_number, int max_frame);
    void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status);
    void featureDetection(Mat img_1, vector<Point2f>& points1);
    double getAbsoluteScale(int frame_id, int sequence_id, double z_cal, std::string txt_path);
    void getCalibData(std::string);   //get camera intrinsic parameters from txt file
    void draw(Mat transM, Mat currImgGry);
};


