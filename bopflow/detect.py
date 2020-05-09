from bopflow.models.yolonet import yolo_v3, BaseNet
from bopflow.const import COCO_DEFAULT_CLASSES


def default_detector(weights_path: str, labels_mapping=dict):
    detector = yolo_v3(
        num_classes=len(labels_mapping), labels_mapping=labels_mapping, just_model=False
    )
    detector.load_weights(weights_path).expect_partial()

    return detector


def coco_yolo_detector(weights_path: str, labels_mapping=COCO_DEFAULT_CLASSES):
    return default_detector(weights_path=weights_path, labels_mapping=labels_mapping)


def coco_yolo_model(saved_model: str):
    detector = BaseNet(labels_mapping=COCO_DEFAULT_CLASSES)
    detector.load_saved_model(saved_model)

    return detector
