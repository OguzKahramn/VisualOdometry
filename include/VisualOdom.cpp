#include "../include/VisualOdom.hpp"
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
#include <cstring>

using namespace std;
using namespace cv;


VisualOdom::VisualOdom(std::string main_path, std::string sequence_number, int max_frame){
  VisualOdom::main_path = main_path;
  VisualOdom::sequence_number = sequence_number;
  VisualOdom::max_frame = max_frame;
  VisualOdom::scale = 1.00;

  std::string img_path = main_path + sequence_number; // + "/image0/%06d.png";
  //cout<< img_path.c_str() << endl;
  std::string calib_txt_file = img_path + "/calib.txt";
  //cout<<calib_txt_file<<endl;
  VisualOdom::getCalibData(calib_txt_file);
  


  sprintf(VisualOdom::imgName1, "%s/image_0/%06d.png", img_path.c_str(),0);
  //cout<< VisualOdom::imgName1<<endl;
  sprintf(VisualOdom::imgName2, "%s/image_0/%06d.png", img_path.c_str(),1);
  VisualOdom::img1 = imread(VisualOdom::imgName1);
  VisualOdom::img2 = imread(VisualOdom::imgName2);

 
  cvtColor(VisualOdom::img1, VisualOdom::img1gry, COLOR_BGR2GRAY);
  cvtColor(VisualOdom::img2, VisualOdom::img2gry, COLOR_BGR2GRAY);
  

  VisualOdom::featureDetection(VisualOdom::img1gry, VisualOdom::points1);
  VisualOdom::featureTracking(VisualOdom::img1gry, VisualOdom::img2gry, VisualOdom::points1, VisualOdom::points2, VisualOdom::status);
  
  VisualOdom::E = findEssentialMat(VisualOdom::points2, VisualOdom::points1, VisualOdom::focal, VisualOdom::pp, RANSAC, 0.999, 1.0, VisualOdom::mask);
  recoverPose(VisualOdom::E, VisualOdom::points2, VisualOdom::points1, VisualOdom::R, VisualOdom::T, VisualOdom::focal, VisualOdom::pp, VisualOdom::mask);
  
  VisualOdom::prevImg = img2gry;
  VisualOdom::prevFeatures = VisualOdom::points2;
  
  VisualOdom::rotM = VisualOdom::R.clone();
  VisualOdom::transM = VisualOdom::T.clone();

  namedWindow( "Grayscale Image Sequence", WINDOW_AUTOSIZE );// Create a window for display.
  namedWindow( "Visual Odometry Trajectory", WINDOW_AUTOSIZE );// Create a window for display.
  VisualOdom::traj = Mat(1000,1000, CV_8UC3, Scalar(255,255,255)).clone();

  for (int i=2; i < VisualOdom::max_frame; i++){
    
    sprintf(VisualOdom::imgName1, "%s/image_0/%06d.png", img_path.c_str(), i);
    //std::cout<< VisualOdom::imgName1<<endl;
    VisualOdom::currImg = imread(VisualOdom::imgName1);

    cvtColor(VisualOdom::currImg, VisualOdom::currImgGry, COLOR_BGR2GRAY);
    
    
    VisualOdom::featureTracking(VisualOdom::prevImg, VisualOdom::currImgGry, VisualOdom::prevFeatures, VisualOdom::currFeatures, VisualOdom::status );

    VisualOdom::E = findEssentialMat(VisualOdom::currFeatures, VisualOdom::prevFeatures, VisualOdom::focal, VisualOdom::pp, RANSAC, 0.999, 1.0, VisualOdom::mask);
    recoverPose(VisualOdom::E, VisualOdom::currFeatures, VisualOdom::prevFeatures, VisualOdom::R, VisualOdom::T, VisualOdom::focal, VisualOdom::pp, VisualOdom::mask);
    
    
    VisualOdom::scale = VisualOdom::getAbsoluteScale(i,0, VisualOdom::T.at<double>(2), img_path);
    //cout<< "Scale" << VisualOdom::scale << endl;
    if ((VisualOdom::scale>0.1)&&(VisualOdom::T.at<double>(2) > VisualOdom::T.at<double>(0)) && (VisualOdom::T.at<double>(2) > VisualOdom::T.at<double>(1))) {
      
      VisualOdom::transM = VisualOdom::transM + VisualOdom::scale * (VisualOdom::rotM * VisualOdom::T);
      
      VisualOdom::rotM = VisualOdom::R * VisualOdom::rotM;
    }
    else {
      cout<< "errorrrr......" << endl;
    }

    if (VisualOdom::prevFeatures.size() < MIN_NUM_FEAT)	{
 		  VisualOdom::featureDetection(VisualOdom::prevImg, VisualOdom::prevFeatures);
      
      VisualOdom::featureTracking(VisualOdom::prevImg, VisualOdom::currImgGry, VisualOdom::prevFeatures, VisualOdom::currFeatures, VisualOdom::status );
 	  }
    
    VisualOdom::prevImg = currImgGry.clone();
    VisualOdom::prevFeatures = VisualOdom::currFeatures;
    VisualOdom::draw(VisualOdom::transM, VisualOdom::currImgGry);

  }
  
}
void VisualOdom::featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 

  vector<float> err;					
  Size winSize=Size(21,21);																								
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  int j = 0;
  for( int i=0; i<status.size(); i++)
     {  Point2f pt = points2.at(i- j);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
     		  points1.erase (points1.begin() + (i - j));
     		  points2.erase (points2.begin() + (i - j));
     		  j++;
     	}

     }

}


void VisualOdom::featureDetection(Mat img_1, vector<Point2f>& points1)	{ 
  vector<KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}

double VisualOdom::getAbsoluteScale(int frame_id, int sequence_id, double z_cal, std::string txt_path)	{
  
  string line;
  int i = 0;
  string file = txt_path+"/"+ VisualOdom::sequence_number + ".txt";
  ifstream myfile (file);
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev; 
  if (myfile.is_open())
  {
    
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }
      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
    return 0;
  }
  //cout<< x_prev << " " << y_prev << " " << z_prev << endl;
  return sqrt(pow((x-x_prev),2)+ pow((y-y_prev),2) + pow((z-z_prev),2)) ;

}

void VisualOdom::draw(Mat transM, Mat img1gry){
      
    int x = int(transM.at<double>(0)) + 500;
    int y = int(transM.at<double>(2)) + 100;
    circle(VisualOdom::traj, Point(x, y) ,1, CV_RGB(0,0,255), 2);

    rectangle(VisualOdom::traj, Point(10, 30), Point(850, 60), CV_RGB(255,255,255), cv::FILLED);
    sprintf(text, "Current coordinates: x = %02f m y = %02f m z = %02f m", transM.at<double>(0), transM.at<double>(1), transM.at<double>(2));
    putText(VisualOdom::traj, text, cv::Point(10,50), FONT_HERSHEY_PLAIN, 1, Scalar::all(0), 1, 8);

    imshow( "Grayscale Image Sequence", img1gry);
    imshow( "Visual Odometry Trajectory", VisualOdom::traj );

    waitKey(1);
    }

void VisualOdom::getCalibData(std::string calib_txt_path){
  std::string line;
  ifstream myfile;
  myfile.open(calib_txt_path);
  int counter = 0;
  
  if (myfile.is_open())
  {
    //cout<<"actik"<<endl;
    while ( getline (myfile,line, ' '))
    {
      //cout<<line<<endl;
      if (counter==1){
        VisualOdom::focal = std::stod(line);
      }
      if (counter==3){
        VisualOdom::pp.x= std::stod(line);
      }
      if (counter==7){
        VisualOdom::pp.y= std::stod(line);
      }
      counter++;
      
    }
    myfile.close();
  }
}