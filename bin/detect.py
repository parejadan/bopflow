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

flags.DEFINE_string("weights", "./checkpoints/yolov3.tf",
                    "path to weights file")
flags.DEFINE_boolean("tiny", False, "yolov3 or yolov3-tiny")
flags.DEFINE_string("image", "./data/girl.png", "path to input image")
flags.DEFINE_string("output", "./output.jpg", "path to output image")


def main(_argv):
    yolo = coco_yolo_detector(use_tiny=FLAGS.tiny)
    logging.info("weights loaded")

    img_raw = load_image_file(FLAGS.image)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, DEFAULT_IMAGE_SIZE)

    t1 = time.time()
    detections = yolo.evaluate(img)
    t2 = time.time()
    logging.info("time: {}".format(t2 - t1))

    logging.info("detections:")
    for detected_class in detections:
        logging.info(detected_class)

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, detections)
    cv2.imwrite(FLAGS.output, img)
    logging.info("output saved to: {}".format(FLAGS.output))


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
