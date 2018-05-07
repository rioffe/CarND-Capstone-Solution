#!/usr/bin/env python

import rospy
import tf
from std_msgs.msg import Int32, Float32, Float32MultiArray, Bool
from std_msgs.msg import MultiArrayDimension, MultiArrayDimension
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from itertools import cycle, islice
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MIN_LOOKAHEAD_DIST = 10 # Min distance away in meters for red light to trigger slow down (this is a safety measure in case of latency)
COMFORT_DECEL = 3 # Max deceleration value (expects a positve number)
COMFORT_ACCEL = 5 # Max acceleration value (expects a positve number)
#TARGET_V = 40*1609.34/60/60 # first number is mph followed by conversion to mps

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # TODO: Add other member variables you need below
        self.vehicle_position = None
        self.vehicle_yaw = None
        self.waypoints = None
        # Previous closest waypoint idx to ego
        self.prev_idx = None
        # Ego's current velocity
        self.v = 0
        self.TARGET_V = 0

        # Stop waypoint closest to nearest upcoming traffic light as published by tl_detector.py
        self.stop_waypoint = -1
        # x-y position of stop line closest to nearest upcoming traffic light as published by tl_detector.py
        self.stop_x = -1
        self.stop_y = -1

        self.dbw_enabled = False

        # Publish closest waypoint so tl_dector.py doesn't have to repeat calculation
        self.closest_waypoint_pub = rospy.Publisher('/closest_waypoint', Int32, queue_size=1)
        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # If a red stop light has been detected within a distance based on current speed
        # that allows enough time for a comfortable deceleration, then we publish it
        # so that it doesn't have to be re-calculated in twist_controller.py, where it
        # is used to calculate torque in Newton meters to control braking
        self.stop_a_pub = rospy.Publisher('/stop_a', Float32, queue_size=1)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/target_v', Float32, self.targetv_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Float32MultiArray, self.traffic_cb)
        #Add dbw enabled so closest waypoint finder can reset
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_cb)

        rospy.spin()

    def pose_cb(self, msg):
        rospy.loginfo("pose_cb is called")

        self.vehicle_position = msg.pose.position
        orientation = msg.pose.orientation
        _, _, self.vehicle_yaw = tf.transformations.euler_from_quaternion((orientation.x,
                 orientation.y, orientation.z, orientation.w))

	self.handle_final_waypoints()

    def handle_final_waypoints(self):
        rospy.loginfo("handle_final_waypoints")

        if not self.waypoints or not self.vehicle_position or not self.vehicle_yaw:
            return

        next_wp = self.next_waypoint(self.waypoints, self.vehicle_position, self.vehicle_yaw)

        final_waypoints = islice(cycle(self.waypoints), next_wp, next_wp + LOOKAHEAD_WPS + 1)

        # stop waypoint = -1 if tl_detector.py sees no traffic light or nearest upcoming traffic light is green

        if self.stop_waypoint != -1:
            stop_dist = self.distance(self.waypoints, next_wp, self.stop_waypoint)

            # Time and distance it will take to stop decelerating comfortably at current v
            lookahead_t = self.v / COMFORT_DECEL
            lookahead_dist = 0.5*COMFORT_DECEL*lookahead_t*lookahead_t

            # MIN_LOOKAHEAD_DIST is a buffer of 10m, so no matter the speed ego will slow down for red lights <= 10m away
            if stop_dist <= max(MIN_LOOKAHEAD_DIST, lookahead_dist):
                stop_a, pub_waypoints = self.generate_stop_trajectory(list(final_waypoints), stop_dist)

            # if red light is too far away for relevance, brake=-1 causing a brake value of 0 in twist_controller.py
            # ego maintains TARGET_V
            else:
                stop_a = -1
                pub_waypoints = self.generate_keep_trajectory(list(final_waypoints))

        # if no light in view or green light, brake=-1 causing a brake value of 0 in twist_controller.py
        # ego maintains TARGET_V
        else:
            stop_a = -1
            pub_waypoints = self.generate_keep_trajectory(list(final_waypoints))

        lane = Lane()
        lane.header.frame_id = '/World'
        lane.header.stamp = rospy.Time.now()

        lane.waypoints = pub_waypoints
        self.final_waypoints_pub.publish(lane)
        # Publish stop acceleration for twist_controller.py
        self.stop_a_pub.publish(Float32(stop_a))

    def closest_waypoint(self, waypoints, position):
        rospy.loginfo("closest_waypoint")
        idx = -1
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

        # Only search through all waypoints if we haven't yet found a closest waypoint,
        # otherwise only loop through waypoints up to LOOKAHEAD_WPS ahead of prev idx
        ds = []
        N = len(self.waypoints)
        if self.prev_idx is None:
            [ds.append(dl(self.waypoints[i].pose.pose.position, position)) for i in range(N)]
            idx = np.argmin(ds)

        else:
            [ds.append(dl(self.waypoints[(self.prev_idx+i)%N].pose.pose.position, position)) for i in range(LOOKAHEAD_WPS)]
            idx = (np.argmin(ds) + self.prev_idx) % N

        self.closest_waypoint_pub.publish(idx)
        self.prev_idx = idx

        return idx

    def next_waypoint(self, waypoints, position, yaw):
        wp_idx = self.closest_waypoint(waypoints, position)
        wp_x = waypoints[wp_idx].pose.pose.position.x
        wp_y = waypoints[wp_idx].pose.pose.position.y
        heading = math.atan2(wp_y - position.y, wp_x - position.x)
        angle = abs(yaw - heading)
        angle = min(2 * math.pi - angle, angle);
        if angle > math.pi / 4:
            wp_idx = (wp_idx + 1) % len(waypoints)
        return wp_idx

    def waypoints_cb(self, waypoints):
        rospy.loginfo("waypoints_cb is called")
        self.waypoints = waypoints.waypoints
        self.handle_final_waypoints()

    def targetv_cb(self, msg):
        rospy.loginfo("targetv_cb is called")
        self.TARGET_V = msg.data
        #unblock below to set TARGET_V manually here
        #self.TARGET_V = 35*1609.34/60/60 # first number is mph followed by conversion to mps

    def velocity_cb(self, msg):
        rospy.loginfo("velocity_cb is called")
        self.v = msg.twist.linear.x

    def traffic_cb(self, msg):
        rospy.loginfo("traffic_cb is called")
        self.stop_waypoint = int(msg.data[0])
        self.stop_x = msg.data[1]
        self.stop_y = msg.data[2]

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def dbw_cb(self, msg):
        rospy.loginfo("dbw_cb is called")
        self.dbw_enabled = msg.data
        # Every time we switch from manual to autonomous, the prev_idx resets,
        # allowing to manually turn the car around and still find the closest waypoint
        self.prev_idx = None

    def generate_keep_trajectory(self, waypoints):
        a = COMFORT_ACCEL
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        cur_v = self.v

        for i in range(len(waypoints)):
            wp0 = waypoints[i-1].pose.pose.position
            wp1 = waypoints[i].pose.pose.position
            cur_v = min(self.TARGET_V, math.sqrt(cur_v*cur_v + 2*a*dl(wp0, wp1)))
            self.set_waypoint_velocity(waypoints, i, cur_v)

        return waypoints

    def generate_stop_trajectory(self, waypoints, stop_dist):
        # stop_dist is distance to wp closest to red light, ~3-5m past where ego should stop_a_pub
        # if dist to stop line (i.e., not wp nearest stop line) is less than 3m, stop acceleration
        # is set to 1m/s^2. Besides ensuring that ego stops at or before stop line,this ensures
        # that ego remains still, rather than drift forward when goal_v = 0.
        stop_line_dist = math.sqrt((self.vehicle_position.x-self.stop_x)**2 + (self.vehicle_position.y-self.stop_y)**2)
        if stop_line_dist < 3:
            a = 1
        else:
            # calculate time and acceleration needed to stop before redlight at current v
            t = (2*stop_dist) / (self.v)
            a = min(COMFORT_DECEL, self.v / max(0.001, t))
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        # Set current ego v
        cur_v = self.v

        # Using acceleration calculated above, decrease velocity of following waypoints_cb
        # so ego will reach 0 by red light
        for i in range(len(waypoints)):
            wp0 = waypoints[i-1].pose.pose.position
            wp1 = waypoints[i].pose.pose.position
            cur_v =  math.sqrt(max(0, cur_v*cur_v - 2*a*dl(wp0, wp1)))
            self.set_waypoint_velocity(waypoints, i, cur_v)
        # In addition to waypoints, return a to publish to twist_controller.py
        return a, waypoints

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        N = len(waypoints)
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

        if wp2 >= wp1:
            for i in range(wp1, wp2):
                dist += dl(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
        else:
            for i in range(wp1, N-1):
                dist += dl(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
            for i in range(wp2):
                dist += dl(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
