import numpy as np
import tensorflow as tf


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(
        tf.minimum(box_1[..., 2], box_2[..., 2])
        - tf.maximum(box_1[..., 0], box_2[..., 0]),
        0,
    )
    int_h = tf.maximum(
        tf.minimum(box_1[..., 3], box_2[..., 3])
        - tf.maximum(box_1[..., 1], box_2[..., 1]),
        0,
    )
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


class BBox:
    def __init__(self, x1y1: np.array, x2y2: np.array):
        self.x1y1 = x1y1
        self.x2y2 = x2y2

    @property
    def as_dict(self):
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    @property
    def x_min(self):
        return self.x1y1[0]

    @property
    def y_min(self):
        return self.x1y1[1]

    @property
    def x_max(self):
        return self.x2y2[0]

    @property
    def y_max(self):
        return self.x2y2[1]


class DetectionOutput:
    def __init__(
        self,
        box: np.array,
        score:np.float32,
        class_number: np.int32,
        class_name: str
    ):
        self.box = BBox(x1y1=box[0:2], x2y2=box[2:4])
        self.confidence_score = score
        self.class_number = class_number
        self.class_name = class_name

    @property
    def as_dict(self):
        return {
            "box": self.box.as_dict,
            "confidence_score": self.confidence_score,
            "class_number": self.class_number,
            "class_name": self.class_name,
        }
