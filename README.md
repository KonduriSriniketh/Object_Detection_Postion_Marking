# ROS Perception Pipeline - Object_Detection_Position_Marking
Detect the object using YOLO and mark the postion on the 2d ROS map.
- Model Trained on YOLOv5.
- PyTorch cpp library with CUDA is used to deploy this model

## zed_obj_dec_pos_tag 
**Instructions to use the object detection package (Perception Pipeline)**

- Video showing the object detectected by camera tagged onto the map.



### Requirements to run this package 
- ZED camera
- pytorch
- CUDA
- Already established 2D localization system on the robot (I.e. TF form camera link to map )

### Command to launch file
    roslaunch zed_obj_dec_pos_tag object_detection_zed.launch 
### Notes
  - ***<---No Implementation of Tracking of detected objects when the object is out of the camera frame---->***
  - Two main Python files are present in the src/ folder of this package (DetectorNode.py & VisionPipeline.py)
  - The Launch file runs the DetectorNode.py file in src/ folder
  - The VisionPipeline.py is imported in the DetectorNode.py
  - In DetectorNode.py the zed camera is launched, and the frame is grabbed with the resolution considered and used to evaluate the detections with their bounding boxes and scores.
  - The object detection pipeline in the VisionPipeline.py used the yolov5 inference to detect the objects, their bounding boxes and their scores 
  - In the constructor of the VisionPipeline, **self.weight, self.data** files represent the **weight file** and the **.yaml** file(configuration file with class names) of the AI model used for detection (Change the model based on the specific detection requirements.)
  - After getting the scores, bounding boxes from the VisionPipeline, each bounding box center coordinate is considered to estimate the depth from the camera to find the distance between camera and the object detected.
  - Using the TF tree the object position is found with respect to /map frame and pointed out in the RVIZ map using the markers.
