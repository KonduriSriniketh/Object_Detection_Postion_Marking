#!/usr/bin/env python3
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
import sys
from pathlib import Path
import platform
import rospy
from vision_msgs.msg import *

from utils.plots import Annotator, colors, save_one_box
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class VisionPipieline():

    def __init__(self) -> None:
        self.weights='/home/sutd/ROS2_developments/AI/trained_weights/exp4/weights/best.pt'
        self.data='/home/sutd/ROS2_developments/AI/trained_weights/exp4/Interior_Furniture_20_03_23.yaml'
        self.imgsz=(640, 640)  # inference size (height, width)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det=1000  # maximum detections per image
        self.classes = None # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False
        self.line_thickness=3
        self.device_type= 'cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.dnn = False
        self.half=False
        self.hide_labels=False
        self.hide_conf=False
        self.view_img = True
        print("load model")
        # Load model
        self.device = select_device(self.device_type)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        print("names = ",self.names)

        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup     

        # Initialize CV_Bridge
        self.bridge = CvBridge()   
        
        pass
    
    def run(self, data_img):

        
        im = data_img
        im, im0 = self.preprocess(im)

        # Run inference
        im = torch.from_numpy(im).to(self.model.device) 
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)


        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        
        # Process predictions 
        for i, det in enumerate(pred):

            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
            # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                   


                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # bounding_box = BoundingBox()
                    c = int(cls)

                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    print("label = ", label)
                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()

            if self.view_img:
                if platform.system() == 'Linux':
                    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow("Detection", im0.shape[1], im0.shape[0])
                cv2.imshow("Detection", im0)
                cv2.waitKey(1)  # 1 millisecond


    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0


   

