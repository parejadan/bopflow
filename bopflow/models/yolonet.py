import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy

from bopflow.models.darknet import darknet_conv_upsampling, darknet_conv, darknet
from bopflow.models.utils import (
    DOutput,
    DLabel,
    reshape_lambda,
    reduce_max_lambda,
    boxes_lambda,
    nms_lambda,
)
from bopflow.const import (
    YOLO_MAX_BOXES,
    YOLO_IOU_THRESHOLD,
    YOLO_SCORE_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    COCO_DEFAULT_CLASSES,
)
from bopflow import LOGGER


def yolo_conv(filters: int, name=None):
    def _yolo_conv(x_in):
        if isinstance(x_in, tuple):
            x, inputs = darknet_conv_upsampling(
                x_in=x_in, filters=filters, size=1, up_sampling=2
            )
        else:
            x = inputs = Input(x_in.shape[1:])

        x = darknet_conv(x=x, filters=filters, size=1)
        x = darknet_conv(x=x, filters=filters * 2, size=3)
        x = darknet_conv(x=x, filters=filters, size=1)
        x = darknet_conv(x=x, filters=filters * 2, size=3)
        x = darknet_conv(x=x, filters=filters, size=1)

        return Model(inputs, x, name=name)(x_in)

    return _yolo_conv


def yolo_output(filters: int, anchors, num_classes, name=None):
    def _yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = darknet_conv(x=x, filters=filters * 2, size=3)
        x = darknet_conv(
            x=x, filters=anchors * (num_classes + 5), size=1, batch_norm=False
        )
        x = reshape_lambda(anchors=anchors, num_classes=num_classes + 5)(x)

        return tf.keras.Model(inputs, x, name=name)(x_in)

    return _yolo_output


def yolo_boxes(pred, anchors, num_classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...num_classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, num_classes), axis=-1
    )

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, num_classes):
    boxes, conf, types = [], [], []

    for o in outputs:
        boxes.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        conf.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        types.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(boxes, axis=1)
    confidence = tf.concat(conf, axis=1)
    class_probs = tf.concat(types, axis=1)

    scores = confidence * class_probs
    boxes, scores, num_classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=YOLO_MAX_BOXES,
        max_total_size=YOLO_MAX_BOXES,
        iou_threshold=YOLO_IOU_THRESHOLD,
        score_threshold=YOLO_SCORE_THRESHOLD,
    )

    return boxes, scores, num_classes, valid_detections


def yolo_loss(anchors, num_classes=80, ignore_thresh=0.5):
    def _yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, num_classes
        )
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold

        best_iou = tf.map_fn(
            reduce_max_lambda, (pred_box, true_box, obj_mask), tf.float32
        )
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = (
            obj_mask
            * box_loss_scale
            * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        )
        wh_loss = (
            obj_mask
            * box_loss_scale
            * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        )
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class
        )

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return _yolo_loss


class BaseNet:
    def __init__(self, labels_mapping: dict):
        self.labels_mapping = labels_mapping
        self.model = None

    def load_saved_model(self, saved_model: str):
        loaded = tf.saved_model.load(saved_model)
        self.model = loaded.signatures["serving_default"]

        return self.model

    def load_weights(self, weights_path):
        LOGGER.info(f"Loading weights from {weights_path}")
        return self.model.load_weights(weights_path)

    @property
    def layer_names(self):
        return [layer.name for layer in self.model.layers]

    def get_label_name(self, target_id):
        for label_name, label_id in self.labels_mapping.items():
            if label_id == target_id:
                return label_name

    def evaluate(self, image):
        """
        Returns
        =======
        [
            DOutput(
                box=BBox(),
                confidence_score: float,  # detection confidence score in (0.5, 1.0)
                label=DLabel(),
            ),
            ...
        ]
        """
        detections = []
        boxes, scores, label_ids, detection_count = self.model(image)
        boxes = boxes[0]
        scores = scores[0]
        label_ids = label_ids[0]
        detection_count = detection_count[0].numpy()
        for i in range(detection_count):
            label_id = int(label_ids[i].numpy())
            detections.append(
                DOutput(
                    box=boxes[i].numpy(),
                    score=scores[i].numpy(),
                    label=DLabel(
                        number=label_id, name=self.get_label_name(target_id=label_id)
                    ),
                )
            )
        return detections


class YoloNet(BaseNet):
    def __init__(
        self,
        channels: int,
        anchors: np.array,
        masks: np.array,
        num_classes: int,
        labels_mapping: dict,
        size=None,
        training=False,
    ):
        super().__init__(labels_mapping=labels_mapping)
        self.channels = channels if channels else 3
        self.num_classes = num_classes if num_classes else 80
        self.size = size
        self.training = training

        if not anchors:
            self.anchors = (
                np.array(
                    [
                        (10, 13),
                        (16, 30),
                        (33, 23),
                        (30, 61),
                        (62, 45),
                        (59, 119),
                        (116, 90),
                        (156, 198),
                        (373, 326),
                    ],
                    np.float32,
                )
                / DEFAULT_IMAGE_SIZE
            )
        else:
            self.anchors = anchors
        if not masks:
            self.masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        else:
            self.masks = masks
        self._conv_creator = yolo_conv
        self.set_model()

    def get_input(self):
        return Input([self.size, self.size, self.channels], name="input")

    def get_conv(self, x: tf.Tensor, x_prev: tf.Tensor, filters: int, mask_index: int):
        x_ins = (x, x_prev) if isinstance(x_prev, tf.Tensor) else x

        conv_tensor = self._conv_creator(
            filters=filters, name=f"yolo_conv_{mask_index}"
        )(x_ins)

        output_layer = yolo_output(
            filters=filters,
            anchors=len(self.masks[mask_index]),
            num_classes=self.num_classes,
            name=f"yolo_output_{mask_index}",
        )(conv_tensor)

        return conv_tensor, output_layer

    def get_lambda_boxes(self, output_layer, mask_index: int):
        lambda_instance = boxes_lambda(
            box_func=yolo_boxes,
            anchors=self.anchors[self.masks[mask_index]],
            num_classes=self.num_classes,
            lambda_name=f"yolo_boxes_{mask_index}",
        )

        return lambda_instance(output_layer)

    def get_output(self, boxes: tuple):
        lambda_instance = nms_lambda(
            nms_func=yolo_nms,
            anchors=self.anchors,
            masks=self.masks,
            num_classes=self.num_classes,
            lambda_name="yolo_nms",
        )

        return lambda_instance(boxes)

    def set_model(self):
        x = inputs = self.get_input()

        x_36, x_61, dark_tensor = darknet(name="yolo_darknet")(x)

        conv_0, output_0 = self.get_conv(
            x=dark_tensor, x_prev=None, filters=512, mask_index=0
        )
        conv_1, output_1 = self.get_conv(
            x=conv_0, x_prev=x_61, filters=256, mask_index=1
        )
        conv_2, output_2 = self.get_conv(
            x=conv_1, x_prev=x_36, filters=128, mask_index=2
        )

        if self.training:
            self.model = Model(inputs, (output_0, output_1, output_2), name="yolov3")
        else:
            boxes_0 = self.get_lambda_boxes(output_layer=output_0, mask_index=0)
            boxes_1 = self.get_lambda_boxes(output_layer=output_1, mask_index=1)
            boxes_2 = self.get_lambda_boxes(output_layer=output_2, mask_index=2)
            outputs = self.get_output(boxes=(boxes_0[:3], boxes_1[:3], boxes_2[:3]))
            self.model = Model(inputs, outputs, name="yolov3")

        return self.model


def yolo_v3(
    size=None,
    channels=3,
    anchors=None,
    masks=None,
    num_classes=80,
    labels_mapping=None,
    training=False,
    just_model=True,
):
    network = YoloNet(
        channels=channels,
        anchors=anchors,
        masks=masks,
        num_classes=num_classes,
        labels_mapping=labels_mapping,
        size=size,
        training=training,
    )
    return network.model if just_model else network


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
