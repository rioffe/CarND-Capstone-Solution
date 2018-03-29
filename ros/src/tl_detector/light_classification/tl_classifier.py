from styx_msgs.msg import TrafficLight
from tl_detector import TLDetector
from tl_detector import _load_image_into_numpy_array
import tensorflow as tf

from PIL import Image
import numpy as np

class TLClassifier(object):
    def __init__(self, model_path='light_classification/models/ssd_sim_and_real_24_03_2018'):
    #def __init__(self, model_path='light_classification/models/frozen_sim_inception13'):
    #def __init__(self, model_path='light_classification/models/frozen_real_inception13'):
    #def __init__(self, model_path='light_classification/models/frozen_faster_rcnn_sim_v2'):
    #def __init__(self, model_path='light_classification/models/frozen_faster_rcnn_reallife_v2'):
        self.detector = TLDetector(model_path)

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

        return tl_class


#Simple test
def _main(_):
    flags = tf.app.flags.FLAGS
    image_path = flags.input_image
    model_path = flags.model_path

    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = _load_image_into_numpy_array(image)

    classifier = TLClassifier(model_path)

    classification = classifier.get_classification(image_np)
    if (classification == 0):
      print("RED")
    elif (classification == 1): 
      print("YELLOW")
    elif (classification == 2):
      print("GREEN")
    else:
      print("UNKNOWN")

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input_image', 'input/test.jpg', 'Path to input image')
    flags.DEFINE_string('model_path', 'models/ssd_sim_and_real_24_03_2018', 'Path to output image')

    tf.app.run(main=_main)
