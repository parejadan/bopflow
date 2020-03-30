from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy

from bopflow.models.darknet import (darknet_conv_upsampling, darknet_conv,
                                    darknet, darknet_tiny)
from bopflow.models.utils import broadcast_iou


flags.DEFINE_integer('yolo_max_boxes', 100,
                     'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def yolo_conv(filters: int, name=None):
    def _yolo_conv(x_in):
        if isinstance(x_in, tuple):
            x, inputs = darknet_conv_upsampling(
                x_in=x_in,
                filters=filters,
                size=1,
                up_sampling=2)
        else:
            x = inputs = Input(x_in.shape[1:])

        x = darknet_conv(x=x, filters=filters, size=1)
        x = darknet_conv(x=x, filters=filters * 2, size=3)
        x = darknet_conv(x=x, filters=filters, size=1)
        x = darknet_conv(x=x, filters=filters * 2, size=3)
        x = darknet_conv(x=x, filters=filters, size=1)

        return Model(inputs, x, name=name)(x_in)

    return _yolo_conv


def yolo_conv_tiny(filters: int, name=None):
    def _yolo_conv(x_in):
        if isinstance(x_in, tuple):
            x, inputs = darknet_conv_upsampling(
                x_in=x_in,
                filters=filters,
                size=1,
                up_sampling=2)
        else:
            x = inputs = Input(x_in.shape[1:])
            x = darknet_conv(x=x, filters=filters, size=1)

        return Model(inputs, x, name=name)(x_in)

    return _yolo_conv


def yolo_output(filters: int, anchors, classes, name=None):
    def _yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = darknet_conv(x=x, filters=filters * 2, size=3)
        x = darknet_conv(x=x, filters=anchors * (classes + 5), size=1, batch_norm=False)
        x = Lambda(
            lambda x: tf.reshape(
                x,
                (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)
            )
        )(x)

        return tf.keras.Model(inputs, x, name=name)(x_in)

    return _yolo_output


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    boxes, conf, types = [], [], []

    for o in outputs:
        boxes.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        conf.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        types.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(boxes, axis=1)
    confidence = tf.concat(conf, axis=1)
    class_probs = tf.concat(types, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores,
            (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections


def yolo_loss(anchors, classes=80, ignore_thresh=0.5):
    def _yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return _yolo_loss


class BaseV3Net:
    def __init__(
        self,
        channels: int,
        classes: int,
        size=None,
        training=False,
    ):
        self.channels = channels if channels else 3
        self.classes = classes if classes else 80
        self.size = size
        self.training = training
        self.model = None

    def get_input(self):
        return Input([self.size, self.size, self.channels], name="input")

    def get_conv(self, x: tf.Tensor, x_prev: tf.Tensor, filters: int, mask_index: int):
        x_ins = (x, x_prev) if isinstance(x_prev, tf.Tensor) else x
        x = self._conv_creator(filters=filters, name=f"yolo_conv_{mask_index}")(x_ins)
        output_layer = yolo_output(
            filters=filters,
            anchors=len(self.masks[mask_index]),
            classes=self.classes,
            name=f"yolo_output_{mask_index}"
            )(x)

        return x, output_layer

    def get_lambda_boxes(self, output_layer, mask_index: int):
        anchors = self.anchors[self.masks[mask_index]]
        lambda_instance = Lambda(
            lambda x: yolo_boxes(pred=x, anchors=anchors, classes=self.classes),
            name=f"yolo_boxes_{mask_index}"
        )

        return lambda_instance(output_layer)

    def get_output(self, boxes: tuple):
        lambda_instance = Lambda(
            lambda x: yolo_nms(x, self.anchors, self.masks, self.classes),
            name="yolo_nms"
        )

        return lambda_instance(boxes)


class YOLOTinyNetwork(BaseV3Net):
    def __init__(
        self,
        channels: int,
        anchors: np.array,
        masks: np.array,
        classes: int,
        size=None,
        training=False,
    ):
        super().__init__(size=size, channels=channels, classes=classes, training=training)
        if not anchors:
            self.anchors = np.array([(10, 14), (23, 27), (37, 58),
                                (81, 82), (135, 169),  (344, 319)],
                                np.float32) / 416
        else:
            self.anchors = anchors
        if not masks:
            self.masks = np.array([[3, 4, 5], [0, 1, 2]])
        else:
            self.masks = masks
        self._conv_creator = yolo_conv_tiny

    def get_model(self):
        x = inputs = self.get_input()

        x_8, x = darknet_tiny(name="yolo_darknet")(x)

        x, output_0 = self.get_conv(x=x, x_prev=None, filters=256, mask_index=0)

        x, output_1 = self.get_conv(x=x, x_prev=x_8, filters=128, mask_index=0)

        if self.training:
            self.model = Model(inputs, (output_0, output_1), name="yolov3")
        else:
            boxes_0 = self.get_lambda_boxes(output_layer=output_0, mask_index=0)
            boxes_1 = self.get_lambda_boxes(output_layer=output_1, mask_index=1)
            outputs = self.get_output(boxes=(boxes_0[:3], boxes_1[:3]))
            self.model = Model(inputs, outputs, name="yolov3_tiny")

        return self.model


class YOLONetwork(BaseV3Net):
    def __init__(
        self,
        channels: int,
        anchors: np.array,
        masks: np.array,
        classes: int,
        size=None,
        training=False,
    ):
        super().__init__(size=size, channels=channels, classes=classes, training=training)
        if not anchors:
            self.anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
        else:
            self.anchors = anchors
        if not masks:
            self.masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        else:
            self.masks = masks
        self._conv_creator = yolo_conv

    def get_model(self):
        x = inputs = self.get_input()

        x_36, x_61, x = darknet(name="yolo_darknet")(x)

        x, output_0 = self.get_conv(x=x, x_prev=None, filters=512, mask_index=0)
        x, output_1 = self.get_conv(x=x, x_prev=x_61, filters=256, mask_index=1)
        x, output_2 = self.get_conv(x=x, x_prev=x_36, filters=128, mask_index=2)

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
    classes=80,
    training=False,
    use_tiny=False,
    just_model=True,
):
    yolo_network = YOLOTinyNetwork if use_tiny else YOLONetwork
    network = yolo_network(
        channels=channels,
        anchors=anchors,
        masks=masks,
        classes=classes,
        size=size,
        training=training,
    )
    if just_model:
        return network.get_model()
    else:
        return network
