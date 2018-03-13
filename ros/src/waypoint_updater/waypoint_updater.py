#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import PoseStamped
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


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.vehicle_position = None
        self.vehicle_yaw = None
        self.waypoints = None
        self.prev_idx = None

        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        rospy.loginfo("pose_cb is called")

        self.vehicle_position = msg.pose.position
        orientation = msg.pose.orientation
        _, _, self.vehicle_yaw = tf.transformations.euler_from_quaternion((orientation.x,
                 orientation.y, orientation.z, orientation.w))

	self.handle_final_waypoints()

    def handle_final_waypoints(self):
        rospy.loginfo("handle_final_waypoints")

        if not self.waypoints:
            return

        if not self.vehicle_position or not self.vehicle_yaw:
            return

        next_wp = self.next_waypoint(self.waypoints,
            self.vehicle_position, self.vehicle_yaw)

        final_waypoints = islice(cycle(self.waypoints), next_wp, next_wp + LOOKAHEAD_WPS + 1)

        lane = Lane()
        lane.header.frame_id = '/World'
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = list(final_waypoints)
        self.final_waypoints_pub.publish(lane)

    def closest_waypoint(self, waypoints, position):
        idx = -1
        min_distance = 100000
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

        #Only search through all waypoints if we haven't yet found a closest waypoing,
        #otherwise only loop through waypoints up to LOOKAHEAD_WPS ahead of prev idx
        ds = []
        N = len(self.waypoints)
        if self.prev_idx is None:
            [ds.append(dl(self.waypoints[i].pose.pose.position, position)) for i in range(N)]
            idx = np.argmin(ds)

        else:
            [ds.append(dl(self.waypoints[(self.prev_idx+i)%N].pose.pose.position, position)) for i in range(LOOKAHEAD_WPS)]
            idx = (np.argmin(ds) + self.prev_idx) % N

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
        # TODO: Implement
        rospy.loginfo("waypoints_cb is called")
        self.waypoints = waypoints.waypoints
        self.handle_final_waypoints()

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
