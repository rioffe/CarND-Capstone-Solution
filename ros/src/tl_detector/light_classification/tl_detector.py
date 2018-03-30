import tensorflow as tf
import numpy as np

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import Image


class TLDetector(object):
    def __init__(self, model_path):
        path_to_ckpt = model_path + '/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.6

        self.detection_session = tf.Session(graph=self.detection_graph, config=config)

        # Get handles to input and output tensors
        ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = self.detection_graph.get_tensor_by_name(
                        tensor_name)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')


    def run_inference_for_single_image(self, image):
        # Run inference
        output_dict = self.detection_session.run(self.tensor_dict,
                feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        return output_dict


def _load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


def _draw_bounding_box_on_image_array(image,
        ymin,
        xmin,
        ymax,
        xmax,
        color='red',
        thickness=4,
        display_str_list=None):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(image_pil)
    im_width, im_height = image_pil.size

    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom),
        (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin

    np.copyto(image, np.array(image_pil))


def _visualize_boxes_and_labels_on_image(image,
        boxes,
        classes,
        scores,
        min_score_thresh=0.25,
        line_thickness=8):
    category_index = {
                1 : 'Green',
                2 : 'Red',
                3 : 'Yellow',
                4 : 'off'
            }
    color_index = {
                1 : 'green',
                2 : 'red',
                3 : 'yellow',
                4 : 'blue'
            }

    for i in range(20):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            class_name = category_index.get(classes[i], 'N/A')
            display_str = '{}: {}%'.format(class_name, int(100*scores[i]))
            color = color_index.get(classes[i], 'blue')

            _draw_bounding_box_on_image_array(image,
                    box[0], box[1], box[2], box[3],
                    color=color,
                    thickness=line_thickness,
                    display_str_list=[display_str,])

    return image


def _main(_):
    flags = tf.app.flags.FLAGS
    image_path = flags.input_image
    output_path = flags.output_image
    model_path = flags.model_path

    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = _load_image_into_numpy_array(image)
    # Actual detection.
    detector = TLDetector(model_path)
    output_dict = detector.run_inference_for_single_image(image_np)

    # print output_dict
    # Visualization of the results of a detection.
    _visualize_boxes_and_labels_on_image(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            min_score_thresh=0.2,
            line_thickness=8)
    # store image to output dir
    im = Image.fromarray(image_np)
    im.save(output_path)


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input_image', 'input/test.jpg', 'Path to input image')
    flags.DEFINE_string('output_image', 'output/test.jpg', 'Path to output image')
    flags.DEFINE_string('model_path', 'models/ssd_sim_and_real_24_03_2018', 'Path to output image')

    tf.app.run(main=_main)
