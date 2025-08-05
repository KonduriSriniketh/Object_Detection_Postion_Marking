
import pyrealsense2 as rs
import numpy as np
import cv2

class rsWrapper:
    def __init__(self):

        # Create a pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.detectCamera = False

        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        self.found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
    
    def configure_stream(self):


        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = 2 #1 meter
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale
        print("Aligning to depth..")
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        # self.intr = self.profile.as_video_stream_profile().get_intrinsics()

        

    def capture_depth_color(self):

        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        self.intr = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        #bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        return depth_image, color_image

    def capture_steam(self):

        # Streaming loop
        try:
            while True:
                
                depth_image, color_image = self.capture_depth_color()

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                images = np.hstack((color_image, depth_colormap))

                cv2.namedWindow('Align depth', cv2.WINDOW_NORMAL)
                cv2.imshow('Align depth', images)
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
            print("Closing camera")
            cv2.destroyAllWindows()
            self.pipeline.stop()
        
        return
    
    def getDepth(self, depth, x,y):
        return depth.get_distance(x, y)
    


if __name__ == "__main__":

    obj = rsWrapper()
    obj.configure_stream()
    obj.capture_steam()
    
