from bopflow.models.yolonet import yolo_v3, BaseNet
from bopflow.const import COCO_DEFAULT_CLASSES


def default_detector(weights_path: str, labels_mapping=COCO_DEFAULT_CLASSES):
    """
    Loads COCO model from a weights.tf resource
    """
    detector = yolo_v3(
        num_classes=len(labels_mapping), labels_mapping=labels_mapping, just_model=False
    )
    detector.load_weights(weights_path).expect_partial()

    return detector


def default_model(saved_model: str, labels_mapping=COCO_DEFAULT_CLASSES):
    """
    Loads COCO model from a saved_model.pb resource
    """
    detector = BaseNet(labels_mapping=labels_mapping)
    detector.load_saved_model(saved_model)

    return detector
