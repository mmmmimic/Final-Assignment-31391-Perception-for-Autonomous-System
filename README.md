# Final Project of 31392 - Perception of Autonomous Systems
## Project Goals
- Calibrate and rectify the stereo input.
- Process the input images to detect objects on the conveyor and track them in 3D, even under occlusion.
- Train a machine learning system that can classify unseen images into 3 classes (cups, books and boxes) based either on 2D or 3D data.

## ToDo
- [x] 4/9/2020 Calibrate and Rectify the stereo input
- [x] 4/23/2020 Object Tracking
- [x] 4/24/2020-5/7/2020 Classification
- [x] 5/7/2020-5/10/2020 Report

## Prerequisite
- python == 3.7.4
- opencv == 3.4.2
- imutils == 0.5.3

## Part1: Calibration and Rectification

### process
1. Calibrate the camera
2. Undistort the checkboard images
3. Recalibrate the camera
4. Undistort the conveyor images
5. Rectification

## Part2: Track Object
- update 05.07.2020 Add classification. Now all the commends are written in English.   
See ./scr/main.py.  
In this part, I use BackgroundSubstracker(BS) to detect moving objects in a frame. To initialize the process, one should select two ROI mannually, one is thought to be the entry of the conveyor while the other one is the exit. After that, as illustrated in the fiugres below, a white conveyor model is made.  
![avatar](/pics/entry.jpg)  
![avatar](/pics/exit.jpg)  
![avatar](/pics/conveyor.jpg)  
If the object found by BS is not on the conveyor, it will be bounded by a green box.  
![avatar](/pics/green.jpg)  
Once it enters the entry we gave before, the green box will become a red one. Both the top view and the front view of the object are shown, since we need to TRACK THE OBJECT IN 3D. TO fetch the top view of the object, we need to use binocular vision method.  
![avatar](/pics/view.jpg)  
If the object is under occlusion, predict it according to the path history.  
![avatar](/pics/predict.jpg)  


 










