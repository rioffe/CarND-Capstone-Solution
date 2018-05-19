import rospy
from styx_msgs.msg import TrafficLight
from tl_detector import TLDetector
from tl_detector import _load_image_into_numpy_array
import tensorflow as tf

from PIL import Image
import numpy as np

class TLClassifier(object):
    def __init__(self, model_path='light_classification/models/ssd_sim_and_real_30_03_2018',
                       isSimulator=False):
        self.isSimulator = isSimulator
        # Primary Model
        if isSimulator:
            self.detector = TLDetector('light_classification/models/frozen_faster_rcnn_sim_v2')
        else:
            self.detector = TLDetector(model_path)

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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        tl_class_index = {
                1 : TrafficLight.GREEN,
                2 : TrafficLight.RED,
                3 : TrafficLight.YELLOW,
                4 : TrafficLight.UNKNOWN
                }

        output_dict = self.detector.run_inference_for_single_image(image)

        tl_class = TrafficLight.UNKNOWN
        if output_dict['detection_scores'][0] >= 0.4:
            tl_class = tl_class_index.get(output_dict['detection_classes'][0],
                    TrafficLight.UNKNOWN)

        if self.isSimulator:
            rospy.loginfo('Marcus says: {} with {}'.format(self.get_state_string(tl_class), output_dict['detection_scores'][0]))
        else:
            rospy.loginfo('Yuri says: {} with {}'.format(self.get_state_string(tl_class), output_dict['detection_scores'][0]))
       
        return tl_class

# Simple test
def _main(_):
    flags = tf.app.flags.FLAGS
    image_path = flags.input_image
    model_path = flags.model_path
    simulator = flags.simulator

    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = _load_image_into_numpy_array(image)

    classifier = TLClassifier(model_path, simulator)

    classification = classifier.get_classification(image_np)
    state_s = classifier.get_state_string(classification)
    print(state_s)

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input_image', 'input/test.jpg', 'Path to input image')
    flags.DEFINE_string('model_path', 'models/frozen_faster_rcnn_reallife_v2', 'Path to the second model')
    flags.DEFINE_bool('simulator', False, 'Whether image is coming from a simulator or not')

    tf.app.run(main=_main)
