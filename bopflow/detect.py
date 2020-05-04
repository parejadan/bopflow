from bopflow.models.yolonet import yolo_v3, BaseNet
from bopflow.const import COCO_DEFAULT_CLASSES


def default_detector(
    weights_path: str,
    labels_mapping=dict,
    use_tiny=False,
):
    detector = yolo_v3(
        num_classes=len(labels_mapping),
        labels_mapping=labels_mapping,
        use_tiny=use_tiny,
        just_model=False,
    )
    detector.load_weights(weights_path).expect_partial()

    return detector


def coco_yolo_detector(
    weights_path: str,
    labels_mapping=COCO_DEFAULT_CLASSES,
    use_tiny=False,
):
    return default_detector(
        weights_path=weights_path,
        labels_mapping=labels_mapping,
        use_tiny=use_tiny,
    )
