import numpy as np
import argparse
import tensorflow as tf

from bopflow.models.yolonet import yolo_v3
from bopflow.models.yolonet import default_detector
from bopflow.iomanage import load_darknet_weights
from bopflow import LOGGER


def export_tf_weights(num_classes, weights_path, output_path):
    yolo = yolo_v3(num_classes=num_classes)
    yolo.summary()
    LOGGER.info("model created")

    load_darknet_weights(yolo, weights_path)
    LOGGER.info("weights loaded")

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    yolo(img)
    LOGGER.info("sanity check passed")

    yolo.save_weights(f"{output_path}/weights.tf")
    LOGGER.info("weights saved")


def export_model(weights_path, output_path):
    yolo = default_detector(weights_path=weights_path)
    LOGGER.info("tf weights model loaded")
    yolo.model.summary()
    LOGGER.info("saving model")
    tf.keras.models.save_model(model=yolo.model, filepath=f"{output_path}/1")


def main(args):
    if args.output_format == "tf":
        export_tf_weights(
            num_classes=args.num_classes,
            weights_path=args.input,
            output_path=args.output_path,
        )
    elif args.output_format == "model":
        export_model(weights_path=args.input, output_path=args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert yolo3 darknet weights to tf yolov3 weights"
    )
    parser.add_argument("-input", help="path to network weights to convert")
    parser.add_argument(
        "--num_classes", default=80, help="number of classes in the model"
    )
    parser.add_argument(
        "--output-format",
        default="tf",
        choices={"tf", "model"},
        help="output format to save parsed weights",
    )
    parser.add_argument(
        "--output-path",
        default="./checkpoints/yolov3",
        help="directory to output converted result into",
    )
    args = parser.parse_args()
    main(args)
