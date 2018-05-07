#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32, Float32MultiArray
from std_msgs.msg import MultiArrayDimension, MultiArrayDimension
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np

# For now state is ground truth, so no need to have a cnt threshold
STATE_COUNT_THRESHOLD = 0

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = None
        self.has_image = False # we don't have image yet

        self.pose_wp_idx = None
        self.tl_wp_idx = [] # Waypoint indices of traffic lights
        self.tl_xy = [] # Stop line positions of traffic lights

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.bridge = CvBridge()
        self.use_simulator_classifier = rospy.get_param('~on_simulator')
        rospy.loginfo("Is on simulator? %s" , self.use_simulator_classifier)
        self.light_classifier = TLClassifier(isSimulator = self.use_simulator_classifier)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.state_count = 0

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Float32MultiArray, queue_size=15)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        # Add closest waypoint subscriber to receive current closest waypoint from waypoint WaypointUpdater
        sub4 = rospy.Subscriber('/closest_waypoint', Int32, self.closest_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        N = len(self.waypoints)
        # Waypoints are only loaded once so at boot find closest waypoint idx of each traffic light stop line

        for x, y in self.config['stop_line_positions']:
            ds = []
            [ds.append(math.sqrt((x-self.waypoints[i].pose.pose.position.x)**2 + (y-self.waypoints[i].pose.pose.position.y)**2)) for i in range(N)]
            best_idx = np.argmin(ds)
            self.tl_wp_idx.append(best_idx)
            self.tl_xy.append([x, y])

    def closest_cb(self, msg):
        self.pose_wp_idx = msg.data

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        
        # Every time waypoint updater finds new closest waypoint, re-calculate location
        # of nearest stop line, waypoint closest to nearest stop line, and state of nearest light
        closest_tl_xy, light_wp, state = self.process_traffic_lights()

        if state == TrafficLight.GREEN:
            light_wp = -1

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        # Publish nearest waypoint and x-y coords of stop line so waypoint updater can slow if necessary
        red_light_pub = Float32MultiArray()
        red_light_pub.layout.dim.append(MultiArrayDimension())
        red_light_pub.layout.dim[0].label = "length"
        red_light_pub.layout.dim[0].size = 3
        red_light_pub.layout.dim[0].stride = 3
        red_light_pub.layout.data_offset = 0
        red_light_pub.data = [light_wp, closest_tl_xy[0], closest_tl_xy[1]]
        self.upcoming_red_light_pub.publish(red_light_pub)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        image = np.asanyarray(cv_image)

        # Get classification
        return self.light_classifier.get_classification(image)

    def get_state_string(self, state):
        if (state == 0):
            state_s = "RED"
        elif (state == 1):
            state_s = "YELLOW"
        elif (state == 2):
            state_s = "GREEN"
        else:
            state_s = "UNKNOWN"

        return state_s

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            list (float): x,y coordinates of nearest traffic light stopline

        """
        closest_tl_wp_idx = 0

        # This assumes ego always travels around loop in start direction. Should be fixed to use Yuri's calculation from waypoint_updater.py.
        closest_tl_wp_idx = min(self.tl_wp_idx)
        closest_tl_xy = self.tl_xy[np.argmin(self.tl_wp_idx)]
        if (self.pose_wp_idx):
            for i in range(len(self.tl_wp_idx)):
                if self.tl_wp_idx[i] > self.pose_wp_idx:
                    closest_tl_wp_idx = self.tl_wp_idx[i]
                    closest_tl_xy = self.tl_xy[i]
                    break

        # We now have x,y position of stopline of closest traffic light.
        # Initially, rather than use camera img and classifier, we can get ground truth state of that light from the simulator.
        stop_x = closest_tl_xy[0]
        stop_y = closest_tl_xy[1]
        state = TrafficLight.UNKNOWN
        if (self.lights):
            n_lights = len(self.lights)
            ds = []
            [ds.append(math.sqrt((stop_x - self.lights[i].pose.pose.position.x)**2 + (stop_y - self.lights[i].pose.pose.position.y)**2)) for i in range(n_lights)]
            if (self.use_simulator_classifier):
                groundtruth = self.lights[np.argmin(ds)].state
                rospy.loginfo('groundtruth is {}'.format(self.get_state_string(groundtruth)))
            
            state = self.get_light_state(self.lights[np.argmin(ds)])
            rospy.loginfo('state is {}'.format(self.get_state_string(state)))

        return closest_tl_xy, closest_tl_wp_idx, state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
