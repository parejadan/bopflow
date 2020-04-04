import time
import cv2
import numpy as np
import tensorflow as tf
import argparse

from bopflow.detect import coco_yolo_detector
from bopflow.transform import transform_images
from bopflow.const import DEFAULT_IMAGE_SIZE
from bopflow.iomanage import draw_outputs, load_image_file
from bopflow import LOGGER


def main(args):
    yolo = coco_yolo_detector(weights_path=args.weights_path, use_tiny=args.tiny)
    LOGGER.info("detector loaded")

    img_raw = load_image_file(args.image)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, DEFAULT_IMAGE_SIZE)

    t1 = time.time()
    detections = yolo.evaluate(img)
    t2 = time.time()
    LOGGER.info("time: {}".format(t2 - t1))

    LOGGER.info("detections:")
    for result in detections:
        LOGGER.info(result.as_dict)

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, detections)
    cv2.imwrite(args.output, img)
    LOGGER.info("output saved to: {}".format(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performce once off image detection using yolov3 default detector"
    )
    parser.add_argument(
        "-image", dest="image", help="image file to perform detection on"
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="pass if you want to perform detection with tiny network",
    )
    parser.add_argument(
        "--weights-path",
        dest="weights_path",
        default="./checkpoints/yolov3.tf",
        help="path to network weights to use for detection",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default="output.jpg",
        help="filepath to output detection result",
    )
    args = parser.parse_args()

    main(args)
