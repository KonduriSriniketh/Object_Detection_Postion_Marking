#!/usr/bin/env python3

import rospy
import torch
from VisionPipeline import VisionPipieline
import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import TransformListener, Buffer, buffer_interface
from tf2_geometry_msgs import tf2_geometry_msgs
from zed_obj_dec_pos_tag.msg import *
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, check_dataset, check_requirements,
                           check_yaml, clean_str, cv2, is_colab, is_kaggle, segments2boxes, unzip_file, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
import time
from pathlib import Path
from threading import Thread
 
class DetectorNode():
    def __init__(self):

        self.marker_topic        = "object_markers"
        self.bounding_box_topic  = "detection_bbox"
        self.depth_image_topic   = "depth_image"
        self.maker_frame_id      = "camera_link"
        self.object_topic        = "objects"
        self.marker_life_time    = 500.0

        self.marker_pub = rospy.Publisher('marker_topic', MarkerArray, queue_size=1)
        self.detection_publish   = rospy.Publisher('depth_image_topic', arrbbox, queue_size=1)
        self.object_pub          = rospy.Publisher('object_topic', ObjectArray, queue_size=1)
        self.source = str(0)
        
        rospy.init_node('talker', anonymous=True)
        self.rate = rospy.Rate(5)
        
    def run(self):
        # Create a Camera object
        zed = sl.Camera()
        rate = rospy.Rate(10)

        new_width = 640
        new_height = 640
        new_image_size = sl.Resolution(new_width, new_height)

        dataset = LoadStreams(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
        init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("exit")
            exit(1)

        # Create and set RuntimeParameters after opening the camera
        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  
        # Setting the depth confidence parameters
        runtime_parameters.confidence_threshold = 100
        runtime_parameters.textureness_confidence_threshold = 100

        camera_info = zed.get_camera_information()
        intrinsics = camera_info.calibration_parameters.left_cam

        print(f"Intrinsic matrix: \n{intrinsics.fx} 0 {intrinsics.cx}\n0 {intrinsics.fy} {intrinsics.cy}\n0 0 1")

        

        i = 0
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()
        
        mirror_ref = sl.Transform()
        mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
        tr_np = mirror_ref.m

        B = VisionPipieline()


        while not rospy.is_shutdown():
        # A new image is available if grab() returns SUCCESS
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve left image
                zed.retrieve_image(image, sl.VIEW.LEFT, resolution=new_image_size)
                # Retrieve depth map. Depth is aligned on the left image
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH, resolution=new_image_size)
                # Retrieve colored point cloud. Point cloud is aligned on the left image.
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            
                bgr_image = image.get_data()
                image_ocv = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)

                # B.run(image_ocv)
                
                self.rate.sleep()
                
            sys.stdout.flush()

    # Close the camera
        zed.close()
        pass

    def convert_depth_pixel_to_metric_coordinate(self, pixel_x, pixel_y, Depth, Intrinsics):

        x_m = (pixel_x - Intrinsics.cx) * Depth / Intrinsics.fx
        y_m = (pixel_y - Intrinsics.cy) * Depth / Intrinsics.fy
        z_m = Depth
        return np.array([x_m, y_m, z_m])
    
    def populate_marker_msg(self, marker_array, frame_id, objects_array, names):
        
        counter  = 0
        marker_array.markers.clear()
        for p in objects_array:

            poseIn = PoseStamped()
            poseIn.header.frame_id = "camera_link"
            poseIn.header.stamp = rospy.Time.now()
            poseIn.pose.position.x    = p.x
            poseIn.pose.position.y    = p.y
            poseIn.pose.position.z    = p.z
            poseIn.pose.orientation.x = 0.0
            poseIn.pose.orientation.y = 0.0
            poseIn.pose.orientation.z = 0.0
            poseIn.pose.orientation.w = 1.0

            marker = Marker()
            marker_text = Marker()

            if frame_id != "camera_link":

                poseOut = PoseStamped()

                if self.trasnformPoint(poseIn, poseOut, frame_id):
                    
                    marker.id = counter
                    marker_text.id = 100 + counter

                    marker.type = Marker.CYLINDER
                    marker_text.type = Marker.TEXT_VIEW_FACING

                    marker.action = Marker.ADD
                    marker_text.action = Marker.ADD

                    # Marker for object
                    marker.header.frame_id = poseOut.header.frame_id
                    marker.header.stamp = poseOut.header.stamp
                    marker.pose = poseOut.pose

                    # Marker for text
                    marker_text.header.frame_id = poseOut.header.frame_id
                    marker_text.header.stamp = poseOut.header.stamp
                    marker_text.pose.orientation = poseOut.pose.orientation
                    marker_text.pose.position = poseOut.pose.position
                    marker_text.pose.position.z = poseOut.pose.position.z + 0.5  # offset the text display over z

                    # Add colors
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0

                    marker_text.color.r = 1.0
                    marker_text.color.g = 1.0
                    marker_text.color.b = 0.0
                    marker_text.color.a = 1.0

                    # add scale
                    marker.scale.x = 0.5
                    marker.scale.y = 0.5
                    marker.scale.z = 0.5

                    marker_text.scale.x = 0.5
                    marker_text.scale.y = 0.5
                    marker_text.scale.z = 0.5

                    marker_text.text = names[counter]
                    marker_text.lifetime = rospy.Duration(1)
                    marker.lifetime = rospy.Duration(1)

                    # print("marker loop true = \n",marker )
                    # print("marker text loop true = \n",marker_text )
                    marker_array.markers.append(marker)
                    marker_array.markers.append(marker_text)
                    counter += 1

            else:
                marker.header.frame_id = poseIn.header.frame_id
                marker.header.stamp = poseIn.header.stamp
                marker.pose = poseIn.pose
                marker.id = counter
                marker_text.id = 100 + counter
                marker.type = Marker.CYLINDER
                marker_text.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD
                marker_text.action = Marker.ADD
                # Marker for text
                marker_text.header.frame_id = poseIn.header.frame_id
                marker_text.header.stamp = poseIn.header.stamp
                marker_text.pose.orientation = poseIn.pose.orientation
                marker_text.pose.position = poseIn.pose.position
                marker_text.pose.position.z = poseIn.pose.position.z + 0.5  # offset the text display over z
                # Add colors
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0   
                marker_text.color.r  = 1.0
                marker_text.color.g  = 1.0
                marker_text.color.b  = 0.0
                marker_text.color.a  = 1.0
                #add scale
                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 0.5
                marker_text.scale.x = 0.5
                marker_text.scale.y = 0.5
                marker_text.scale.z = 0.5
                marker_text.text        = names[counter]
                marker_text.lifetime    = rospy.Duration(self.marker_life_time)
                marker.lifetime         = rospy.Duration(self.marker_life_time)
                # print("marker loop false = \n",marker )
                # print("marker text loop false = \n",marker_text )
                marker_array.markers.append(marker)
                marker_array.markers.append(marker_text)
                counter += 1

        # print("marker_array in loop = \n", marker_array)
        return

    def populate_object_msg(self, object_array, objects_array, names):
        counter = 0
        object_array.object_array.clear()
        object_array.header.stamp = rospy.Time.now()
        object_array.header.frame_id = "camera_link"

        for p in objects_array:
            poseIn = PoseStamped()
            obj = Object()

            obj.pose.position.x = p.x
            obj.pose.position.y = p.y
            obj.pose.position.z = p.z
            obj.pose.orientation.x = 0.0
            obj.pose.orientation.y = 0.0
            obj.pose.orientation.z = 0.0
            obj.pose.orientation.w = 1.0
            obj.name.data = names[counter]
            object_array.object_array.append(obj)
            counter += 1

        return object_array

    def trasnformPoint(self, poseIn, poseOut, to_frame):

        tfBuffer = Buffer()
        tfListener = TransformListener(tfBuffer)
        transformStamped = TransformStamped()

        try:
            transformStamped = tfBuffer.lookup_transform("map", "camera_link", rospy.Time(0))
            poseOut = tfBuffer.transform(poseIn, to_frame, rospy.Duration(0.0))
            return True
        except TransformException as ex:
            rospy.logwarn(str(ex))
            # rospy.sleep(1.0)
            return False


if __name__ == "__main__":
    try:
        A = DetectorNode()
        print("initialised")
        A.run()
    except rospy.ROSInterruptException:
        pass