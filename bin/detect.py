import time
import tensorflow as tf
import argparse

from bopflow.models.yolonet import default_detector, default_model
from bopflow.transform.image import transform_images
from bopflow.const import DEFAULT_IMAGE_SIZE
from bopflow.iomanage import load_image_file
from bopflow import LOGGER


def main(args):
    yolo = default_model(saved_model=args.weights_path)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performce once off image detection using yolov3 default detector"
    )
    parser.add_argument(
        "-image", dest="image", help="image file to perform detection on"
    )
    parser.add_argument(
        "--weights-path",
        dest="weights_path",
        default="./checkpoints/yolov3.tf",
        help="path to network weights to use for detection",
    )
    args = parser.parse_args()

    main(args)
