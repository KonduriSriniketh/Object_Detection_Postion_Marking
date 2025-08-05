#!/usr/bin/env python3
import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
from detect import VisionPipieline

class DetectorNode():
    def __init__(self):

        # Create a Camera object
        zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
        init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Create and set RuntimeParameters after opening the camera
        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  
        # Setting the depth confidence parameters
        runtime_parameters.confidence_threshold = 100
        runtime_parameters.textureness_confidence_threshold = 100

        new_width = 640
        new_height = 640
        new_image_size = sl.Resolution(new_width, new_height)

        i = 0
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()\
        
        mirror_ref = sl.Transform()
        mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
        tr_np = mirror_ref.m

        B = VisionPipieline()

        while (1):
        # A new image is available if grab() returns SUCCESS
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve left image
                zed.retrieve_image(image, sl.VIEW.LEFT, resolution=new_image_size)
                # Retrieve depth map. Depth is aligned on the left image
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                # Retrieve colored point cloud. Point cloud is aligned on the left image.
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            
                bgr_image = image.get_data()
                image_ocv = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)

                B.run(image_ocv)
                # cv2.namedWindow("Input")
                # cv2.imshow("Input", image_ocv)  
                # cv2.waitKey(1)
                
            sys.stdout.flush()

    # Close the camera
        zed.close()

        
if __name__ == "__main__":
    A = DetectorNode()