import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from bopflow.detect import coco_yolo_detector
from bopflow.transform import transform_images
from bopflow.const import DEFAULT_IMAGE_SIZE, COCO_DEFAULT_CLASSES
from bopflow.iomanage import draw_outputs, load_random_tfrecord_dataset, load_image_file

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    yolo = coco_yolo_detector(use_tiny=FLAGS.tiny)
    logging.info('weights loaded')

    logging.info('classes loaded')

    if FLAGS.tfrecord:
        img_raw, _ = load_random_tfrecord_dataset(FLAGS.tfrecord, FLAGS.classes)
    else:
        img_raw = load_image_file(FLAGS.image)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, DEFAULT_IMAGE_SIZE)

    t1 = time.time()
    boxes, scores, classes, nums = yolo.evaluate(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(COCO_DEFAULT_CLASSES[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), COCO_DEFAULT_CLASSES)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
