from bopflow.models.yolonet import yolo_v3, BaseNet
from bopflow.const import COCO_DEFAULT_CLASSES


def default_detector(
    weights_path: str, class_count: str, class_names=[], use_tiny=False
):
    detector = yolo_v3(
        num_classes=class_count,
        class_names=class_names,
        use_tiny=use_tiny,
        just_model=False,
    )
    detector.load_weights(weights_path).expect_partial()

    return detector


def coco_yolo_detector(weights_path: str, use_tiny=False):
    class_count = len(COCO_DEFAULT_CLASSES)
    return default_detector(
        weights_path=weights_path,
        class_count=class_count,
        class_names=COCO_DEFAULT_CLASSES,
        use_tiny=use_tiny,
    )


def coco_yolo_model(saved_model: str):
    # TODO - fix batchnormalizer serialization with mode.save
    # N10tensorflow3VarE does not exist
    detector = BaseNet(class_names=COCO_DEFAULT_CLASSES)
    detector.load_saved_model(saved_model)

    return detector
