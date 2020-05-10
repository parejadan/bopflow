import argparse
import cv2

from bopflow.iomanage import load_random_tfrecord_dataset
from bopflow.training.visualize import draw_outputs
from bopflow.models.utils import DOutput, DLabel
from bopflow import LOGGER


def extract_box_and_labels(features):
    detections = []
    for x in features:
        box = x[0:4].numpy()
        label_id = x[4].numpy()
        # label_text = x[5].numpy()
        if sum(box) == 0:
            # box coordinates with sum of 0 indicate we've reached
            # the end of defined labels
            break
        detections.append(
            DOutput(box=box, score=1.0, label=DLabel(number=label_id, name=label_id))
        )

    return detections


def main(tfrecord):
    LOGGER.info("Loading tfrecord dataset")
    raw_image, features = load_random_tfrecord_dataset(tfrecord)

    LOGGER.info("Parsing tfrecord row")
    detections = extract_box_and_labels(features=features)
    LOGGER.info("Labels:")
    for result in detections:
        LOGGER.info(result.as_dict)

    img = cv2.cvtColor(raw_image.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, detections)
    cv2.imwrite("output.png", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load random row from *.tfrecord and writes to .png to visualizing annotations"
    )
    parser.add_argument("-tfrecord", help="tfrecord file to load dataset from")
    args = parser.parse_args()

    main(tfrecord=args.tfrecord)
