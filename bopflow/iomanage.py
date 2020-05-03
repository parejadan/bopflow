import numpy as np
import tensorflow as tf

from bopflow.const import DEFAULT_IMAGE_SIZE, YOLOV3_LAYER_LIST, YOLOV3_TINY_LAYER_LIST
from bopflow.transform.records import tfrecord_row_decode


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, "rb")

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


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def load_tfrecord_dataset(file_pattern, size=DEFAULT_IMAGE_SIZE):
    LINE_NUMBER = -1

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: tfrecord_row_decode(x, size))


def load_random_tfrecord_dataset(file_pattern):
    dataset = load_tfrecord_dataset(file_pattern=file_pattern)
    dataset = dataset.shuffle(512)
    img_raw, label = next(iter(dataset.take(1)))

    return img_raw, label


def load_image_file(image_filepath):
    return tf.image.decode_image(open(image_filepath, "rb").read(), channels=3)
