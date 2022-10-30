# VisualOdometry
VisualOdometry

This repo is for learning visual odometry. Visual odometry is basically getting features from image sequences and track them in upcoming frames, if feature matches are perfect, then we need at least 5 points to compute rotation and translation of the camera. But most of the cases feature matches are not perfect therefore we need another algorithm which is called RANSAC. 

In order to download and run the program:

```
git clone https://github.com/OguzKahramn/VisualOdometry.git
cd VisualOdom
cmake .
make
./VisualOdom
```
Before running the program, you need to have the KITTI dataset. Dataset folder should consist of sequence folders (00, 01, 02, ... etc) and inside of these sequence folders, there should be image_0 folder which stores the image files, calib.txt for camera parameters and the text file as the same name as the sequence number which stores the pose information. 

In ```main.cpp``` file, you need to modify arguments of the VisualOdom object according to your dataset path. The constructor takes 3 arguments path of the dataset, sequence number, and the maximun frame number.

After settings dataset folder as I mentioned before, it is an example how to pass arguments:

``` VisualOdom vo("/home/oguzk/VisualOdom/dataset/", "08", 4000); ```

In this case, 08th sequence will be played and trajectory will be appeared till 4000 frames. 

The 8th sequence trajectory : 

![](https://user-images.githubusercontent.com/60695165/198877187-7cb58160-ce94-4ca4-b3cf-f1a3d6e2baa3.png)


