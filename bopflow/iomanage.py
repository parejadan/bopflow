import numpy as np
import tensorflow as tf
import cv2

from bopflow.const import DEFAULT_IMAGE_SIZE


YOLOV3_LAYER_LIST = [
    "yolo_darknet",
    "yolo_conv_0",
    "yolo_output_0",
    "yolo_conv_1",
    "yolo_output_1",
    "yolo_conv_2",
    "yolo_output_2",
]

YOLOV3_TINY_LAYER_LIST = [
    "yolo_darknet",
    "yolo_conv_0",
    "yolo_output_0",
    "yolo_conv_1",
    "yolo_output_1",
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, "rb")
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith("conv2d"):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and sub_model.layers[
                i + 1
            ].name.startswith("batch_norm"):
                batch_norm = sub_model.layers[i + 1]

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape)
            )
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, "failed to read all data"
    wf.close()


def draw_outputs(img, detections):
    wh = np.flip(img.shape[0:2])
    for detected_class in detections:
        bounding_box = detected_class["bounding_box"]
        detection_confidence = detected_class["detection_confidence"]
        class_name = detected_class["class_name"]

        x1y1 = tuple((bounding_box[0:2] * wh).astype(np.int32))
        x2y2 = tuple((bounding_box[2:4] * wh).astype(np.int32))

        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(
            img,
            "{} {:.4f}".format(class_name, detection_confidence),
            x1y1,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 0, 255),
            2,
        )
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def load_tfrecord_dataset(file_pattern, class_file: str, size=DEFAULT_IMAGE_SIZE):
    LINE_NUMBER = -1
    class_table = tf.lookup.StaticHashTable(
        tf.lookup.TextFileInitializer(
            class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"
        ),
        -1,
    )

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))


def load_random_tfrecord_dataset(file_pattern, class_file):
    dataset = load_tfrecord_dataset(file_pattern=file_pattern, class_file=class_file)
    dataset = dataset.shuffle(512)
    img_raw, label = next(iter(dataset.take(1)))

    return img_raw, label


def load_image_file(image_filepath):
    return tf.image.decode_image(open(image_filepath, "rb").read(), channels=3)
