import numpy as np
import argparse

from bopflow.models.yolonet import yolo_v3
from bopflow.iomanage import load_darknet_weights
from bopflow import LOGGER


def main(args):
    yolo = yolo_v3(num_classes=args.num_classes, use_tiny=True)
    yolo.summary()
    LOGGER.info("model created")

    load_darknet_weights(yolo, args.weights_path, args.tiny)
    LOGGER.info("weights loaded")

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    LOGGER.info("sanity check passed")

    yolo.save_weights(args.output)
    LOGGER.info("weights saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert yolo3 darknet weights to tf yolov3 weights"
    )
    parser.add_argument(
        "-num_classes", dest="image", help="number of classes in the model"
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="pass if you want to perform conversion with tiny network",
    )
    parser.add_argument(
        "--weights-path",
        dest="weights_path",
        help="path to network weights to use for detection",
    )
    parser.add_argument(
        "--output", dest="output", help="filepath for converted weights output"
    )
    args = parser.parse_args()

    args.output = args.output if args.output else "./checkpoints/yolov3.tf"
    args.weights_path = (
        args.weights_path if args.weights_path else "./data/darknet.weights"
    )
    args.num_classes = args.num_classes if args.num_classes else 80

    main(args)
