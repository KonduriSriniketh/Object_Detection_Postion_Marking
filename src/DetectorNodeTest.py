#!/usr/bin/env python3

import rospy
import torch
from VisionPipeline import VisionPipieline
import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
from rsWrapper import rsWrapper
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
import csv
 
class DetectorNode():
    def __init__(self, 
                 cam_type = "realsense"
                 ):

        self.cam_type = cam_type
        self.marker_topic        = "object_markers"
        self.bounding_box_topic  = "detection_bbox"
        self.depth_image_topic   = "depth_image"
        self.maker_frame_id      = "camera_link"
        self.object_topic        = "objects"
        self.marker_life_time    = 0.0

        self.marker_pub = rospy.Publisher('marker_topic', MarkerArray, queue_size=1)
        self.detection_publish   = rospy.Publisher('depth_image_topic', arrbbox, queue_size=1)
        self.object_pub          = rospy.Publisher('object_topic', ObjectArray, queue_size=1)
        
        rospy.init_node('zed_detector', anonymous=True)
        self.rate = rospy.Rate(5)
        
    def run(self):
        # Create a Camera object
        if self.cam_type == "zed":
            zed = sl.Camera()
            zed.close()
            rate = rospy.Rate(10)

            new_width = 640
            new_height = 640
            new_image_size = sl.Resolution(new_width, new_height)

            # Create a InitParameters object and set configuration parameters
            init_params = sl.InitParameters()
            init_params.sdk_verbose = True
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
            init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = 30

            zed_opened = False
            exit_counter = 1

            while not zed_opened :
            # Open the camera
                if exit_counter > 5:
                    break
                err = zed.open(init_params)
                if err != sl.ERROR_CODE.SUCCESS:
                    print('Re-connecting..!, attempt', exit_counter, 'out of 5')
                    exit_counter+=1
                    time.sleep(0.5)
                else:
                    zed_opened = True

            if not zed_opened:
                print("exit")
                exit(1)
            # Create and set RuntimeParameters after opening the camera
            runtime_parameters = sl.RuntimeParameters()
            #runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  
            # Setting the depth confidence parameters
            runtime_parameters.confidence_threshold = 100
            runtime_parameters.texture_confidence_threshold = 100

            camera_info = zed.get_camera_information()
            #zed_intrinsics = camera_info.calibration_parameters.left_cam
            zed_intrinsics = camera_info.camera_configuration.calibration_parameters.left_cam
            print(f"Intrinsic matrix: \n{zed_intrinsics.fx} 0 {zed_intrinsics.cx}\n0 {zed_intrinsics.fy} {zed_intrinsics.cy}\n0 0 1")



            i = 0
            image = sl.Mat()
            depth = sl.Mat()
            point_cloud = sl.Mat()

            mirror_ref = sl.Transform()
            mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
            tr_np = mirror_ref.m

        elif self.cam_type == "realsense":
            realsense_ob = rsWrapper()
            realsense_ob.configure_stream()
            image_ocv, depth_ocv = realsense_ob.capture_depth_color()
            pass
        else:
            print("no camera seleted!")

        B = VisionPipieline()

        with open('/home/sutd/meerkat_ros_ws/src/meerkat_stack/meerkat_launch//launch/human_positions.csv', mode='w', newline='') as file:
            
            print("csv opened")
            csv_writer = csv.writer(file)
            csv_writer.writerow(['x_points', 'y_points', 'density'])

            while not rospy.is_shutdown():

                if (self.cam_type == "zed"):
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
                        depth_ocv = depth.get_data()
                    
                        #cv2.imshow("depth",depth_ocv)
                        #cv2.waitKey()
                elif (self.cam_type == "realsense"):
                    depth_ocv, image_ocv = realsense_ob.capture_depth_color()
                    # image_ocv = cv2.resize(image_ocv_raw, (640, 640))
                    # depth_ocv = cv2.resize(depth_ocv_raw, (640, 640))
                    # print("Shape of image:   ", image_ocv.shape)
                    # print("realsense grabbing image")



                B.run(image_ocv)
                object_names = B.getNames()
                scores = B.getScore()
                objects = B.getObjects()
                detected_boxes = B.getBoundingBoxes()
                print("------------------------------------------------------------------------------------------------------------")
                print("detected_boxes = ", detected_boxes)
                print("object_names =", object_names)
                print("scores =", scores)
                print("objects =", objects)
                print("----------")
                objects_array = []
                arr_bbox_msg = arrbbox()
                
                i = 0
                for o in objects:
                    p = Point()
                    bbox_msg = bbox()
                    if(self.cam_type == "zed"):
                        depth_value = depth.get_value(o[0],o[1])[1]
                    else:
                        depth_value = realsense_ob.getDepth(depth_ocv, o[0],o[1])
                    print("depth_value = ", depth_value)
                    #a = np.array([o[0],o[1], depth_value])
                    #print("p = ", p)
                    if (self.cam_type == "zed"):
                        a = self.convert_depth_pixel_to_metric_coordinate(o[0], o[1], depth_value, zed_intrinsics)
                        p.x, p.y, p.z = a[2], -1*a[0], -1*a[1] 
                    elif(self.cam_type == "realsense"):
                        a = self.convert_depth_pixel_to_metric_coordinate(o[0], o[1], depth_value, realsense_ob.intr)
                        p.x, p.y, p.z = a[2], -1*a[0], -1*a[1] 
                        
                    # return np.array([x_m, y_m, z_m])

                    print("p.x, p.y, p.z =",p.x, p.y, p.z )
                    if (math.isnan(p.x) or  math.isnan(p.x) or math.isnan(p.x)):
                        continue
                    
                    if (math.isinf(p.x) or  math.isinf(p.x) or math.isinf(p.x)):
                        continue
                    
                    objects_array.append(p)
                    csv_writer.writerow([p.x, p.y, 1])
                    continue
                    bbox_msg.class_name = object_names[i]
                    bbox_msg.score = scores[i]

                    try:
                        if len(detected_boxes[i]) == 4:
                            for j in range(4):
                                bbox_msg.bbox.append(detected_boxes[i][j])
                                arr_bbox_msg.detected.append(bbox_msg)
                    except Exception as e:
                        print("Unknown exception: ", e)
                        i += 1 
                    print("objects_array......... = \n", objects_array)

                if len(objects_array) > 0:
                    marker_array = MarkerArray()
                    object_array = ObjectArray()
                    self.populate_marker_msg(marker_array, self.maker_frame_id, objects_array, object_names)
                    #print ("marker_array = \n",marker_array)
                    self.populate_object_msg(object_array, objects_array, object_names)
                    print("marketr")
                    print("marker_array =",marker_array)
                    self.marker_pub.publish(marker_array)
                    self.object_pub.publish(object_array)
                    self.detection_publish.publish(arr_bbox_msg)
                
                arr_bbox_msg.detected.clear()
                # bridge = CvBridge()
                # if image_pub_.get_num_connections() > 0:
                #     depth_alligned = #your depth_alligned image
                #     try:
                #         img_msg = bridge.cv2_to_imgmsg(depth_alligned, "mono16")
                #         img_msg.header.stamp = rospy.Time.now()
                #         img_msg.header.frame_id = "/depth_alligned"
                #         image_pub_.publish(img_msg)
                #     except CvBridgeError as e:
                #         print(e)
                # self.rate.sleep()
                
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

                print("Hello!")

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
                    marker.color.r = np.random.uniform(0,1,1)[0]
                    marker.color.g = np.random.uniform(0,1,1)[0]
                    marker.color.b = np.random.uniform(0,1,1)[0]
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
                print("Hey!")
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
                marker.color.r = np.random.uniform(0,1,1)[0]
                marker.color.g = np.random.uniform(0,1,1)[0]
                marker.color.b = np.random.uniform(0,1,1)[0]
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
                print(poseIn)
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


# class DataStreams():
#     def __init__(self, sources='file.streams', img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
#             torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
#             self.mode = 'stream'
#             self.img_size = img_size
#             self.stride = stride
#             self.vid_stride = vid_stride  # video frame-rate stride
#             sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
#             n = len(sources)
#             self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
#             print("enumerate(sources) = ",enumerate(sources))
#             for i, s in enumerate(sources):  # index, source
#                 # Start thread to read frames from video stream
#                 print("@@@@@@@@@@@@@@@")
#                 print("i = ", i)
#                 print("s = ", s)
#                 print("@@@@@@@@@@@@@@@")
#                 # Start thread to read frames from video stream
#                 st = f'{i + 1}/{n}: {s}... '

#                 if s == 0:

#                     zed = sl.Camera()
#                     init_params = sl.InitParameters()
#                     init_params.camera_resolution = sl.RESOLUTION.HD720
#                     init_params.camera_fps = 30
#                     init_params.camera_device_id = s    
#                     err = zed.open(init_params)
#                     if err != sl.ERROR_CODE.SUCCESS:
#                         zed.close()
#                         raise Exception(f"{st}Failed to open {s}")  

#                     w, h = zed.get_resolution().width, zed.get_resolution().height
#                     self.frames[i] = float('inf')  # infinite stream fallback
#                     self.fps[i] = init_params.camera_fps  # 30 FPS fallback

#                     _, self.imgs[i] = zed.retrieve_image()  # guarantee first frame
#                     self.threads[i] = Thread(target=self.update, args=([i, zed, s]), daemon=True)
#                     print(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
#                     self.threads[i].start()
#                     print.info('') 


#                     # check for common shapes
#                     s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
#                     self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
#                     self.auto = auto and self.rect
#                     self.transforms = transforms  # optional
#                     # if not self.rect:
#                         #     LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

#     def update(self, i, zed, stream):
#     # Read stream `i` frames in daemon thread
#         n, f = 0, self.frames[i]  # frame number, frame array
#         while zed.grab() == sl.ERROR_CODE.SUCCESS and n < f:
#             n += 1
#             if n % self.vid_stride == 0:
#                 success, im = zed.retrieve_image()
#                 if success:
#                     self.imgs[i] = im.get_data()
#                 else:
#                     print.warning('WARNING ⚠️ Video stream unresponsive, please check your ZED camera connection.')
#                     self.imgs[i] = np.zeros_like(self.imgs[i])
#                     init_params = sl.InitParameters()
#                     init_params.camera_resolution = sl.RESOLUTION.HD720
#                     init_params.camera_fps = 30
#                     init_params.camera_device_id = stream
#                     err = zed.close()
#                     if err != sl.ERROR_CODE.SUCCESS:
#                         LOGGER.warning('WARNING ⚠️ Error closing ZED camera.')
#                     err = zed.open(init_params)
#                     if err != sl.ERROR_CODE.SUCCESS:
#                         LOGGER.warning('WARNING ⚠️ Error reopening ZED camera.')
#                 time.sleep(0.0)  # wait time

#     def __iter__(self):
#         self.count = -1
#         return self
    
#     def __next__(self):
#         self.count += 1
#         if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
#             cv2.destroyAllWindows()
#             raise StopIteration
#         im0 = self.imgs.copy()
#         if self.transforms:
#             im = np.stack([self.transforms(x) for x in im0])  # transforms
#         else:
#             im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
#             im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
#             im = np.ascontiguousarray(im)  # contiguous
#         return self.sources, im, im0, None, ''
    
if __name__ == "__main__":
    try:
        A = DetectorNode("zed")
        print("initialised")
        A.run()
    except rospy.ROSInterruptException:
        pass