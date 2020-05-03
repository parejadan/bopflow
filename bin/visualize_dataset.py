import argparse
import cv2
import numpy as np

from bopflow.iomanage import load_tfrecord_dataset
from bopflow.transform.image import transform_images
from bopflow.training.utils import draw_outputs
from bopflow.models.utils import DOutput, DLabel
from bopflow import LOGGER


def extract_box_and_labels(features):
    detections = []
    for x in features:
        box = x[0:4].numpy()
        label_id = x[4].numpy()
        #label_text = x[5].numpy()
        if sum(box) == 0:
            # box coordinates with sum of 0 indicate we've reached
            # the end of defined labels
            break
        detections.append(
            DOutput(
                box=box,
                score=1.0,
                label=DLabel(number=label_id, name=label_id),
            )
        )

    return detections


def main(tfrecord, class_filepath):
    LOGGER.info("Loading classes file")
    class_names = [c.strip() for c in open(class_filepath).readlines()]

    LOGGER.info("Loading tfrecord dataset")
    dataset = load_tfrecord_dataset(tfrecord, class_filepath)
    dataset = dataset.shuffle(512)

    LOGGER.info("Parsing tfrecord row")
    for raw_image, features in dataset.take(1):
        detections = extract_box_and_labels(features=features)
        LOGGER.info("Labels:")
        for result in detections:
            LOGGER.info(result.as_dict)

        img = cv2.cvtColor(raw_image.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, detections)
        cv2.imwrite("output.png", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads random dataset from *.tfrecord and writes to .png file for visualizing annotations"
    )
    parser.add_argument("-tfrecord", help="tfrecord file to load dataset from")
    parser.add_argument("-classes-file", help="Path to *.name file containing tfrecord classes")
    args = parser.parse_args()

    main(tfrecord=args.tfrecord, class_filepath=args.classes_file)
