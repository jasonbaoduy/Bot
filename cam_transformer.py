#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from simple_camera_transformer.cfg import TransformerConfig

class image_transformer:
    def __init__(self):
        rospy.init_node('cam_transformer', anonymous=True)

        sub_topic = rospy.get_param('~sub_topic', '/cam_pub/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/image_transformed')

        self.bridge = CvBridge()
        self.config = None
        self.prev_angular_z = 0.0

        self.cfg_srv = Server(TransformerConfig, self.configCallback)

        self.image_sub = rospy.Subscriber(sub_topic, Image, self.image_callback)
        self.pub = rospy.Publisher(pub_topic, Image, queue_size=10)
        self.cmd_pub = rospy.Publisher('/prizm/twist_controller/twist_cmd', Twist, queue_size=10)

        self.obstacle_detected = False
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        rospy.spin()

    def configCallback(self, cfg, level):
        self.config = cfg
        return cfg

    def applyTransform(self, frame):
        if self.config is None:
            return frame

        frame = cv2.resize(frame, (0, 0), fx=self.config.resize, fy=self.config.resize, interpolation=cv2.INTER_AREA)

        if self.config.enable_color_correct:
            frame = (frame * self.config.cc_alpha + self.config.cc_beta).astype(np.uint8)

        if self.config.enable_sharpen:
            frame = cv2.addWeighted(frame, 1.0 + self.config.sharp_weight, frame, -1.0 * self.config.sharp_weight, 0).astype(np.uint8)

        return frame

    def detect_blue_line(self, frame):
        height, width, _ = frame.shape
        roi = frame[int(height * 0.6):, :]  # Bottom 40% of image

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Mask to exclude white (low saturation, high value)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Blue color ranges
        lower_blue1 = np.array([85, 20, 150])
        upper_blue1 = np.array([120, 80, 255])

        lower_blue2 = np.array([100, 60, 130])
        upper_blue2 = np.array([130, 150, 255])

        mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
        mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)

        combined_mask = cv2.bitwise_or(mask1, mask2)

        # Remove white pixels from the blue mask
        blue_mask_no_white = cv2.bitwise_and(combined_mask, cv2.bitwise_not(white_mask))

        return blue_mask_no_white

    def compute_angular_velocity(self, mask):
        height, width = mask.shape
        center_x = width // 2
        force = 0

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0.0

        for x in xs:
            offset = x - center_x
            force += offset / float(center_x)

        force /= len(xs)
        return -force * 0.5

    def smooth_angular_velocity(self, angular_z, alpha=0.3):
        smoothed = alpha * angular_z + (1 - alpha) * self.prev_angular_z
        self.prev_angular_z = smoothed
        return np.clip(smoothed, -1.0, 1.0)

    def scan_callback(self, scan_msg):
        front_ranges = scan_msg.ranges[len(scan_msg.ranges)//2 - 15 : len(scan_msg.ranges)//2 + 15]
        close_ranges = [r for r in front_ranges if 0.1 < r < 0.25]
        self.obstacle_detected = len(close_ranges) > 0

    def image_callback(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        mask = self.detect_blue_line(cv_image)
        angular_z = self.compute_angular_velocity(mask)
        blue_pixel_count = np.count_nonzero(mask)

        twist = Twist()

        if self.obstacle_detected:
            twist.linear.x = 0.0
            twist.angular.z = 0.9  # Slow spin to avoid obstacle
        elif blue_pixel_count < 100:
            # Search mode — no line visible
            twist.linear.x = 0.0
            twist.angular.z = 0.25  # Fast spin to find line
        else:
            # Line-following mode with smoothing
            angular_z = self.smooth_angular_velocity(angular_z)

            if abs(angular_z) > 0.5:
                twist.linear.x = 0.25  # Slow down for sharp turns
            else:
                twist.linear.x = 0.35
            twist.angular.z = angular_z

        self.cmd_pub.publish(twist)
        self.pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding='passthrough'))

if __name__ == '__main__':
    IT = image_transformer()
