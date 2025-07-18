#!/usr/bin/env python3

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
import rospy
import numpy as np
from behaviors import behavioral_coordination
from tf.transformations import euler_from_quaternion

class behavior_based_control():
    def __init__(self):
        self.current_xyT = np.zeros((3), dtype=float)
        self.obstacles = np.empty((0, 2), dtype=float)
        self.path_points = None  # N x 2 numpy array

        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_cb)
        self.odom_sub  = rospy.Subscriber("/odom_mcl", Odometry, self.odom_cb)
        self.path_sub  = rospy.Subscriber("/path", Path, self.path_cb)
        self.cmd_pub   = rospy.Publisher("/cmd_vel", Twist, queue_size=5)

    def path_cb(self, path_msg: Path):
        points = []
        for pose in path_msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            points.append([x, y])
        if points:
            self.path_points = np.array(points)

    def odom_cb(self, odom_msg: Odometry):
        self.current_xyT[0] = odom_msg.pose.pose.position.x
        self.current_xyT[1] = odom_msg.pose.pose.position.y
        quat = [
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        ]
        rpy = euler_from_quaternion(quat)
        self.current_xyT[2] = rpy[2]

    def send_motion_cmd(self, xy_vector: np.ndarray):
        dist = np.linalg.norm(xy_vector)
        angle = np.arctan2(xy_vector[1], xy_vector[0])
        t_msg = Twist()

        t_msg.linear.x = 0.2 * dist
        t_msg.angular.z = angle

        if np.abs(angle) > np.pi / 2.0:
            t_msg.linear.x = 0.05

        t_msg.linear.x = min(t_msg.linear.x, 1.0)
        t_msg.angular.z = np.clip(t_msg.angular.z, -2.0, 2.0)

        self.cmd_pub.publish(t_msg)

    def laser_cb(self, laser_msg: LaserScan):
        angles = np.array([laser_msg.angle_min + i * laser_msg.angle_increment for i in range(len(laser_msg.ranges))])
        ranges = np.array(laser_msg.ranges)
        valid = np.where((ranges > 0.05) & (ranges < laser_msg.range_max))[0]
        if valid.size == 0:
            return

        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        self.obstacles = np.vstack((x, y)).T

        if self.path_points is not None:
            try:
                motion_xy = behavioral_coordination(
                    path=self.path_points,
                    current_xyT=self.current_xyT,
                    obstacle_xy=self.obstacles
                )
                self.send_motion_cmd(motion_xy)
            except Exception as e:
                rospy.logwarn(f"Behavior coordination error: {e}")

if __name__ == '__main__':
    rospy.init_node("behavior_based_control")
    BBC = behavior_based_control()
    rospy.spin()
