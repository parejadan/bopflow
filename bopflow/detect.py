from bopflow.models.yolonet import yolo_v3


def default_detector(weights_path: str, class_count: str, use_tiny=False):
    detector = yolo_v3(classes=class_count, use_tiny=use_tiny, just_model=False)
    detector.load_weights(weights_path).expect_partial()

    return detector
