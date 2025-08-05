#!/usr/bin/env python3
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
import pyzed.sl as sl
import sys
from pathlib import Path
import platform
import rospy
import math
import time
from vision_msgs.msg import *

from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread


from utils.plots import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, check_dataset, check_requirements,
                           check_yaml, clean_str, cv2, is_colab, is_kaggle, segments2boxes, unzip_file, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class VisionPipieline():

    def __init__(self):
        self.weights='/home/sutd/meerkat_ros_ws/src/zed_obj_dec_pos_tag/src/models/yolov5m.pt'
        self.data='/home/sutd/meerkat_ros_ws/src/zed_obj_dec_pos_tag/src/models/yolov5m.yaml'
        self.imgsz=(640, 640)  # inference size (height, width)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det=1000  # maximum detections per image
        self.classes = None # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False
        self.line_thickness=3
        self.device_type= '0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.dnn = False
        self.half=False
        self.hide_labels=False
        self.hide_conf=False
        self.view_img = True

        self.source = str(0)
        self.vid_stride=1
        self.objects = []
        self.object_names = []
        self.scores = []
        self.detected_boxes = []

        print("load model")
        # Load model
        self.device = select_device(self.device_type)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        print("names = ",self.names)

        
          
        
        pass
    
    def run(self):

        # Initialize CV_Bridge
        
        

        dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        bs = len(dataset)  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup     

        for path, im, im0s, vid_cap, s in dataset:
            print("Srini")
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred = self.model(im, augment=False, visualize=False)


            pred = non_max_suppression(
                pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
            )

            self.objects.clear()
            self.object_names.clear()
            print("==============")
            # Process predictions 
            for i, det in enumerate(pred):
                im0, frame = im0s[i].copy(), dataset.count
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                self.detected_boxes.clear()
                self.scores.clear()
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
                        left   = float(xyxy[0])
                        top    = float(xyxy[1])
                        right  = float(xyxy[2])
                        bottom = float(xyxy[3])
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        bounding_box = [left, top, right -left, bottom - top]
                        self.detected_boxes.append(bounding_box)
                        self.scores.append(float(conf))
                        self.objects.append([left +  (right - left)/2, top +(bottom - top)/2 ])
                        self.object_names.append(self.names[c])
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
    
    def getObjects(self):
        return self.objects
    
    def getNames(self):
        return self.object_names
    
    def getScore(self):
        return self.scores
    
    def getBoundingBoxes(self):
        return self.detected_boxes

   

class LoadStreams:

    def __init__(self, sources='file.streams', img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
            torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
            self.mode = 'stream'
            self.img_size = img_size
            self.stride = stride
            self.vid_stride = vid_stride  # video frame-rate stride
            sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
            n = len(sources)
            self.sources = [clean_str(x) for x in sources]
            self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
            print("enumerate(sources) = ",enumerate(sources))

            self.zed = sl.Camera()

            new_width = self.img_size[0]
            new_height = self.img_size[1]
            self.new_image_size = sl.Resolution(int(new_width), int(new_height))
            self.init_params = sl.InitParameters()
            self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
            self.init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
            self.init_params.camera_resolution = sl.RESOLUTION.HD720
            self.init_params.camera_fps = 30
            
            for i, s in enumerate(sources):  # index, source
                # Start thread to read frames from video stream
                print("@@@@@@@@@@@@@@@")
                print("i = ", i)
                print("s = ", s)
                print("@@@@@@@@@@@@@@@")
                # Start thread to read frames from video stream
                st = f'{i + 1}/{n}: {s}... '
                s = eval(s) if s.isnumeric() else s
                

                err = self.zed.open(self.init_params)
                if err != sl.ERROR_CODE.SUCCESS:
                    print("exit")
                    self.zed.close()
                    raise Exception(f"{st}Failed to open {s}")  
                print("initialised")
                # Create and set RuntimeParameters after opening the camera
                self.runtime_parameters = sl.RuntimeParameters()
                self.runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  
                # Setting the depth confidence parameters
                self.runtime_parameters.confidence_threshold = 100
                self.runtime_parameters.textureness_confidence_threshold = 100
                self.camera_info = self.zed.get_camera_information()
                self.intrinsics = self.camera_info.calibration_parameters.left_cam
                print(f"Intrinsic matrix: \n{self.intrinsics.fx} 0 {self.intrinsics.cx}\n0 {self.intrinsics.fy} {self.intrinsics.cy}\n0 0 1")
                zed_image = sl.Mat()
                
                fps = self.zed.get_current_fps()
                self.frames[i] = float('inf')  # infinite stream fallback
                self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
                if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve left image
                    self.zed.retrieve_image(zed_image, sl.VIEW.LEFT, resolution=self.new_image_size) # guarantee first frame
                    bgr_image = zed_image.get_data()
                    self.imgs[i] = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)
                    print("size of self.imgs[i] = ", self.imgs[i])
                    w, h= (new_width, new_height)
                self.threads[i] = Thread(target=self.update, args=([i, self.zed, s]), daemon=True)
                print(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
                self.threads[i].start()
            LOGGER.info('') 


                # check for common shapes
            s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
            self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
            self.auto = auto and self.rect
            self.transforms = transforms  # optional
            if not self.rect:
                    LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, zed, stream):
    # Read stream `i` frames in daemon thread
        n, f = 0, self.frames[i]  # frame number, frame array
        im = sl.Mat()
        while zed.is_opened() and n < f:
            n += 1
            zed.grab(self.runtime_parameters)
            print("n  outside loop= ", n)
            if n % self.vid_stride == 0:
                
                if (zed.retrieve_image(im, sl.VIEW.LEFT, resolution=self.new_image_size) == sl.ERROR_CODE.SUCCESS):
                    bgr_image = im.get_data()
                    self.imgs[i] = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)
                    print("n  inside loop= ", n)
                else:
                    print.warning('WARNING ⚠️ Video stream unresponsive, please check your ZED camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    init_params = sl.InitParameters()
                    init_params.camera_resolution = sl.RESOLUTION.HD720
                    init_params.camera_fps = 30

                    err = zed.close()
                    if err != sl.ERROR_CODE.SUCCESS:
                        LOGGER.warning('WARNING ⚠️ Error closing ZED camera.')
                    err = zed.open(init_params)
                    if err != sl.ERROR_CODE.SUCCESS:
                        LOGGER.warning('WARNING ⚠️ Error reopening ZED camera.')
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self
    
    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        im0 = self.imgs.copy()

        print("length of im0 in loop =", len(im0))
        print("print im0 = \n",im0)
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous
        return self.sources, im, im0, None, ''
    
    def __len__(self):
        return len(self.sources)
    
if __name__ == "__main__":
    try:
        A = VisionPipieline()
        
        A.run()
    except rospy.ROSInterruptException:
        pass
