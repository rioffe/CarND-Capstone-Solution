import tensorflow as tf
import yaml
import os
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_tfrecord', '', 'Path to output TFRecord')
flags.DEFINE_string('output_label_map', '', 'Path to output label map')
flags.DEFINE_string('config_file', '', 'Path to yaml config file')
flags.DEFINE_string('relative_data_path', '', 'Path to date, e.g. rgb/test for bosch test, used instead path from yaml')

FLAGS = flags.FLAGS

FILTER = {
    "Green" : 1,
    "Red" : 2,
    "Yellow" : 3,
    "off" : 4
}

# Not used, keep it to have full list of Bosch dataset labels
LABEL_DICT =  {
    "Green" : 1,
    "Red" : 2,
    "GreenLeft" : 3,
    "GreenRight" : 4,
    "RedLeft" : 5,
    "RedRight" : 6,
    "Yellow" : 7,
    "off" : 8,
    "RedStraight" : 9,
    "GreenStraight" : 10,
    "GreenStraightLeft" : 11,
    "GreenStraightRight" : 12,
    "RedStraightLeft" : 13,
    "RedStraightRight" : 14
    }

def create_tf_example(example, width, height):
    filename = example['path'].encode() # Filename of the image. Empty if image is not from file

    with tf.gfile.GFile(example['path'], 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = 'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in example['annotations']:
	if not box['class'] in FILTER:
            continue

        #if box['occluded'] is False:

        xmins.append(float(box['xmin']) / width)
        xmaxs.append(float(box['xmin'] + box['x_width'])  / width)
        ymins.append(float(box['ymin']) / height)
        ymaxs.append(float(box['ymin'] + box['y_height']) / height)
#	classes_text.append('light'.encode())
#	classes.append(1)
        classes_text.append(box['class'].encode())
        classes.append(int(FILTER[box['class']]))

    tf_example = None

    if classes:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_tfrecord)

    config = FLAGS.config_file

    examples = yaml.load(open(config, 'rb').read())

    total = len(examples)
    print("Total examples: {}".format(total))

    for i,ex in enumerate(examples):
	if not FLAGS.relative_data_path:
            examples[i]['path'] = os.path.abspath(os.path.join(
                 os.path.dirname(config), ex['filename']))
        else:
            examples[i]['path'] = os.path.abspath(os.path.join(
                 os.path.dirname(config), FLAGS.relative_data_path,
                 os.path.basename(ex['filename'])))

    for i, ex in enumerate(examples):
        width = 1368
        height = 1096
        if 'sim_data_capture' in ex['filename']:
            width = 800
            height = 600
        tf_example = create_tf_example(ex, width, height)
	if tf_example:
            writer.write(tf_example.SerializeToString())

        if i % 100 == 0:
            print("Percents processed: {}".format((float(i) / total) * 100))

    writer.close()

    print("Writing label map pbtxt...")

    with open(FLAGS.output_label_map, "w") as f:
       for key, value in FILTER.iteritems():
           f.write('''
item {{
  id: {}
  name: "{}"
}}'''.format(value, key))


if __name__ == '__main__':
    tf.app.run()
