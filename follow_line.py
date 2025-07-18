#!/usr/bin/env python3
import rospy
import cv2
import sys
import math
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class ColorFollower:
    def __init__(self):
        rospy.init_node("follow_line")

        # HSV thresholds (raised S/V lows to reject pale noise)
        self.lower = np.array([ 90,  50,  50], np.uint8)
        self.upper = np.array([140, 255, 255], np.uint8)

        # Linear speed limits
        self.MIN_V       = rospy.get_param("~min_v",      0.25)
        self.MAX_V       = rospy.get_param("~max_v",      0.40)
        self.speed_boost = rospy.get_param("~speed_boost",1.2)

        # Angular control gains & limits
        self.ang_kp_pos     = rospy.get_param("~ang_kp_pos",   0.005)
        self.max_ang_cruise = rospy.get_param("~max_ang_cruise",0.02)
        self.max_ang_turn   = rospy.get_param("~max_ang_turn",  0.2)

        # TURN behavior tweaks
        self.turn_lin       = rospy.get_param("~turn_lin",      self.MIN_V * 1.5)
        self.turn_ang_alpha = rospy.get_param("~turn_ang_alpha",0.5)
        self.prev_ang_z     = 0.0

        # Hysteresis turn thresholds [deg]
        self.TURN_ENTER_DEG = rospy.get_param("~turn_enter_deg",30.0)
        self.TURN_EXIT_DEG  = rospy.get_param("~turn_exit_deg", 20.0)

        # Lookahead fraction before allowing turn
        self.turn_lookahead = rospy.get_param("~turn_lookahead",0.85)

        # Deadband for “centered”
        self.deadband_ratio = rospy.get_param("~deadband",   0.01)

        # Blur kernel size
        self.blur_ksize     = rospy.get_param("~blur_ksize", 5)

        # Speckle‐filter thresholds (lowered for distant detection)
        self.min_contour_area = rospy.get_param("~min_contour_area", 20)
        self.min_mask_ratio   = rospy.get_param("~min_mask_ratio",   0.02)

        # State: TRACKING, TURN, GAP, DEAD_END
        self.state = "TRACKING"

        # ROS pubs/subs
        self.bridge         = CvBridge()
        self.cmd_pub        = rospy.Publisher("/prizm/twist_controller/twist_cmd",
                                               Twist, queue_size=1)
        self.mask_pub       = rospy.Publisher("/follow_line/mask",
                                               Image, queue_size=1)
        self.mask_annot_pub = rospy.Publisher("/follow_line/mask_annotated",
                                               Image, queue_size=1)
        self.annot_pub      = rospy.Publisher("/follow_line/annotated",
                                               Image, queue_size=1)
        rospy.Subscriber("/cam_pub/image_raw", Image,
                         self.image_cb, queue_size=1)

        rospy.loginfo("Listening for images on /cam_pub/image_raw")
        rospy.spin()

    def branch_length(self, mask, x_start, y_start, direction, max_steps=50):
        length = 0
        x, y = int(x_start), int(y_start)
        h, w = mask.shape
        for _ in range(max_steps):
            y -= 1
            x += direction
            if y < 0 or x < 0 or x >= w or mask[y, x] == 0:
                break
            length += 1
        return length

    def image_cb(self, msg):
        # 1) Acquire frame
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        # 2) Blur
        k = self.blur_ksize | 1
        blurred = cv2.GaussianBlur(frame, (k, k), 0)

        # 3) HSV + inRange
        hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # 4) Morphology cleanup
        k5 = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)
        mask = cv2.medianBlur(mask, 5)
        k7 = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k7, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)

        # 5) Contour filter tiny speckles
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(mask)
        for c in cnts:
            if cv2.contourArea(c) > self.min_contour_area:
                cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)
        mask = clean

        # 6) Relative-size threshold → DEAD_END
        h, w = mask.shape
        if cv2.countNonZero(mask) < self.min_mask_ratio * h * w:
            mask[:] = 0
            self.state = "DEAD_END"
            rospy.loginfo("State changed → DEAD_END")
            self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))
            mask_annot = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.mask_annot_pub.publish(
                self.bridge.cv2_to_imgmsg(mask_annot, "bgr8"))
            twist = Twist()
            rospy.loginfo("DEAD_END behavior: spinning until blue returns")
            twist.linear.x  = 0.0
            twist.angular.z = self.max_ang_turn
            self.cmd_pub.publish(twist)
            return

        # publish cleaned mask
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))

        # 7) DEAD_END spinning handler
        if self.state == "DEAD_END":
            twist = Twist()
            mask_annot = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.mask_annot_pub.publish(
                self.bridge.cv2_to_imgmsg(mask_annot, "bgr8"))
            twist.linear.x  = 0.0
            twist.angular.z = self.max_ang_turn
            self.cmd_pub.publish(twist)
            sys.stdout.write(f"\rDEAD_END → spinning\033[K")
            sys.stdout.flush()
            if np.any(mask):
                self.state = "TRACKING"
            return

        # 8) Annotation & twist setup
        annotated  = blurred.copy()
        mask_annot = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        twist      = Twist()

        # 9) Full-contour detection
        full_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # 10) GAP vs DEAD_END if no contours
        if not full_cnts:
            gap_region = mask[0:int(h*0.3), :]
            if np.any(gap_region):
                self.state = "GAP"
            else:
                self.state = "DEAD_END"
                rospy.loginfo("State changed → DEAD_END")
            if self.state == "GAP":
                x_gap = w/2.0
                for y in range(int(h*0.3), -1, -1):
                    cols = np.where(mask[y,:] > 0)[0]
                    if cols.size:
                        x_gap = float(cols.mean())
                        break
                twist.linear.x  = self.MIN_V
                err_gap         = x_gap - (w/2.0)
                twist.angular.z = clamp(
                    -self.ang_kp_pos * err_gap,
                    -self.max_ang_cruise,
                     self.max_ang_cruise)
                status = "gap → seeking"
            else:
                rospy.loginfo("DEAD_END behavior: turning at end")
                twist.linear.x  = 0.0
                twist.angular.z = self.max_ang_turn
                status = "dead end → turning"
        else:
            # 11) TURN vs TRACKING logic (unchanged)...
            cnt       = max(full_cnts, key=cv2.contourArea)
            vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy    = float(vx), float(vy)
            raw_angle = math.degrees(math.atan2(vx, vy)) % 180.0
            angle_rel_vert = raw_angle if raw_angle <= 90.0 else 180.0 - raw_angle

            # draw heading arrow
            cv2.arrowedLine(annotated,
                            (int(x0), int(y0)),
                            (int(x0 + vx*50), int(y0 + vy*50)),
                            (0,255,0), 2)

            # ——— centroid search over entire frame — detect far‑away blue ———
            y_pt = None
            for y in range(h-1, -1, -1):
                cols = np.where(mask[y,:] > 0)[0]
                if cols.size:
                    y_pt = y
                    break
            if y_pt is None:
                ys   = np.where(mask > 0)[0]
                y_pt = int(ys.min()) if ys.size else 0
                cols = np.where(mask[y_pt,:] > 0)[0]

            x_pt    = float(cols.mean()) if cols.size else w/2.0
            err_pos = x_pt - (w/2.0)

            # hysteresis between TURN and TRACKING
            frac_down = y_pt / float(h)
            if (self.state == "TRACKING"
                and angle_rel_vert > self.TURN_ENTER_DEG
                and frac_down > self.turn_lookahead):
                self.state = "TURN"
            elif (self.state == "TURN"
                  and angle_rel_vert < self.TURN_EXIT_DEG):
                self.state = "TRACKING"

            if self.state == "TURN":
                # TURN behavior unchanged...
                left_len  = self.branch_length(mask, x_pt, y_pt, -1)
                right_len = self.branch_length(mask, x_pt, y_pt, +1)
                dir_sign  = -1 if right_len > left_len else 1
                raw_turn  = (angle_rel_vert / 90.0) * self.max_ang_turn
                new_ang   = clamp(dir_sign * raw_turn,
                                  -self.max_ang_turn,
                                   self.max_ang_turn)
                smooth_ang = (self.turn_ang_alpha * self.prev_ang_z +
                              (1.0 - self.turn_ang_alpha) * new_ang)
                twist.linear.x   = clamp(self.turn_lin,
                                         self.MIN_V,
                                         self.MAX_V)
                twist.angular.z  = smooth_ang
                self.prev_ang_z  = smooth_ang
                status = f"turning: {angle_rel_vert:.1f}°"
            else:
                # TRACKING behavior unchanged...
                norm_err   = abs(err_pos) / (w/2.0)
                dist_ratio = 1.0 - (y_pt / float(h))
                align_fac  = 1.0 - norm_err
                raw_speed  = ((self.MIN_V +
                               (self.MAX_V - self.MIN_V)
                               * dist_ratio * align_fac)
                              * self.speed_boost)
                twist.linear.x   = clamp(raw_speed,
                                         self.MIN_V,
                                         self.MAX_V)
                twist.angular.z  = clamp(-self.ang_kp_pos * err_pos,
                                         -self.max_ang_cruise,
                                          self.max_ang_cruise)
                status = "aligned" if abs(err_pos) < w*self.deadband_ratio else \
                         ("left" if err_pos>0 else "right")

        # 12) Publish annotated & mask images, cmd and console update
        self.mask_annot_pub.publish(
            self.bridge.cv2_to_imgmsg(mask_annot, "bgr8"))
        self.annot_pub.publish(
            self.bridge.cv2_to_imgmsg(annotated,  "bgr8"))
        self.cmd_pub.publish(twist)
        sys.stdout.write(
            f"\rcmd → lin.x={twist.linear.x:.3f}  "
            f"ang.z={twist.angular.z:.3f}  {status}\033[K")
        sys.stdout.flush()

if __name__ == "__main__":
    try:
        ColorFollower()
    except rospy.ROSInterruptException:
        pass
