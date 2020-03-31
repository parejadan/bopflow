from bopflow.models.yolonet import yolo_v3
from bopflow.const import COCO_DEFAULT_CLASSES


def default_detector(weights_path: str, class_count: str, use_tiny=False):
    detector = yolo_v3(classes=class_count, use_tiny=use_tiny, just_model=False)
    detector.load_weights(weights_path).expect_partial()

    return detector


def coco_yolo_detector(use_tiny=False):
    class_count = len(COCO_DEFAULT_CLASSES)
    weights_path = "./checkpoints/yolov3.tf"
    return default_detector(weights_path=weights_path, class_count=class_count, use_tiny=use_tiny)
