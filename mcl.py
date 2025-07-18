#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from utils import normalize_angle, normalize_angle_arr, distN, laser_map
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
import pdb

DEFAULT_NUM_PARTICLES=1000
DEFAULT_LASER_RANGE=5.0
DEFAULT_RANDOM_PCT=0.1
MOTION_NOISE = [0.05, 0.05, 0.02]  # Noise in x, y, and theta

class grid2D:
    def __init__(self, minXY, maxXY, width, height):
        self.minXY = np.array(minXY)
        self.maxXY = np.array(maxXY)
        self.width = width
        self.height = height

class monte_carlo_localization():
    def __init__(self):
        self.num_particles = rospy.get_param('~num_particles', DEFAULT_NUM_PARTICLES)
        self.max_distance = rospy.get_param('~laser_range', DEFAULT_LASER_RANGE)
        self.random_pct = rospy.get_param('~random_particle_percentage', DEFAULT_RANDOM_PCT)

        print("Num Particles = %d, Laser Range=%f, Random Percentage=%f" % (self.num_particles, self.max_distance, self.random_pct))
        self.map = grid2D(minXY=[-10, -10], maxXY=[10, 10], width=20, height=20)

        self.particles = self.create_random_particles(self.num_particles)
        self.current_odom_xyT = None
        self.last_odom_xyT = None

        self.odom_sub = rospy.Subscriber('/odom_noisy', Odometry, self.odom_cb, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_cb, queue_size=1)
        self.initialpose_sub = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initialpose_cb)
        self.marker_pub = rospy.Publisher('/mcl_points', Marker, queue_size=5)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=5)

    def create_random_particles(self, howmany):
        P = np.random.uniform([self.map.minXY[0], self.map.minXY[1], -np.pi], 
                              [self.map.maxXY[0], self.map.maxXY[1], np.pi], 
                              (howmany, 3))
        return P

    def initialpose_cb(self, initialpose: PoseWithCovarianceStamped):
        X = initialpose.pose.pose.position.x
        Y = initialpose.pose.pose.position.y
        quat = [initialpose.pose.pose.orientation.x,
                initialpose.pose.pose.orientation.y,
                initialpose.pose.pose.orientation.z,
                initialpose.pose.pose.orientation.w]
        rpy = euler_from_quaternion(quat)
        particle = np.array([X, Y, rpy[2]], dtype=float).reshape((1, 3))
        weights = np.ones(1)
        self.particles = self.importance_sampling(particle, weights, self.num_particles)

    def publish_pose(self, header, robot_pose):
        odom_out = Odometry()
        odom_out.header = header
        odom_out.pose.pose.position.x = robot_pose[0]
        odom_out.pose.pose.position.y = robot_pose[1]
        quat = quaternion_from_euler(0, 0, robot_pose[2])
        odom_out.pose.pose.orientation.x = quat[0]
        odom_out.pose.pose.orientation.y = quat[1]
        odom_out.pose.pose.orientation.z = quat[2]
        odom_out.pose.pose.orientation.w = quat[3]
        self.odom_pub.publish(odom_out)

    def odom_cb(self, odom_msg: Odometry):
        self.last_odom_xyT = self.current_odom_xyT
        self.current_odom_xyT = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            euler_from_quaternion([
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w
            ])[2]
        ])

        if self.last_odom_xyT is not None:
            self.particles = self.update_pose_from_odom()
            robot_pose = np.mean(self.particles, axis=0)
            self.publish_pose(odom_msg.header, robot_pose)
            self.publish_marker_array(odom_msg.header)

    def update_pose_from_odom(self):
        delta = self.current_odom_xyT - self.last_odom_xyT
        noise = np.random.normal(0, MOTION_NOISE, self.particles.shape)
        self.particles += delta + noise
        return self.particles
    
    def importance_sampling(self, particles, weights, num_samples):
        weights /= weights.sum() + 1e-9  # Normalize weights
        indices = np.random.choice(len(particles), size=num_samples, p=weights)
        return particles[indices]

    def calculate_weights(self, laser_dist: np.ndarray, laser_angles: np.ndarray):
        p_weight = np.zeros(self.num_particles)
        for p_idx in range(self.num_particles):
            expected_distance = np.random.uniform(0, self.max_distance, size=len(laser_angles))
            p_weight[p_idx] = np.exp(-np.linalg.norm(laser_dist - expected_distance))
        return p_weight / np.sum(p_weight)  # Normalize weights
        
    def laser_cb(self, laser_msg: LaserScan):
        langles = np.linspace(laser_msg.angle_min, 
                              laser_msg.angle_min + len(laser_msg.ranges) * laser_msg.angle_increment, 
                              len(laser_msg.ranges))
        ldist = np.clip(laser_msg.ranges, 1e-6, self.max_distance)
        p_weight = self.calculate_weights(ldist, langles)

        num_random = int(self.num_particles * self.random_pct)
        num_samples = self.num_particles - num_random
        self.particles[:num_samples, :] = self.importance_sampling(self.particles, p_weight, num_samples)
        self.particles[num_samples:, :] = self.create_random_particles(num_random)
    
    def publish_marker_array(self, header):
        m = Marker()
        m.header = header
        m.header.frame_id = "map"
        m.type = Marker.POINTS
        m.scale.x = 0.02
        m.scale.y = 0.02
        m.id = 0
        m.color.g = 1.0
        m.color.a = 1.0

        for p in self.particles:
            m.points.append(Point(x=p[0], y=p[1]))
        self.marker_pub.publish(m)

if __name__ == '__main__':
    rospy.init_node("monte_carlo_localization")
    MCL = monte_carlo_localization()
    rospy.spin()