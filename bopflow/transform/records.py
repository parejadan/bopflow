import tensorflow as tf

from bopflow.const import YOLO_MAX_BOXES


class FeatureEncode:
    @classmethod
    def int64_list(cls, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @classmethod
    def int64(cls, values):
        return FeatureParse.int64_list([values])

    @classmethod
    def bytes_list(cls, values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    @classmethod
    def bytes(cls, values):
        return FeatureParse.bytes_list([values])

    @classmethod
    def float_list(cls, values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))


class TFRecordLabels:
    width = "image/width"
    height = "image/height"
    filename = "image/filename"
    source_id = "image/source_id"
    encoded = "image/encoded"
    file_format = "image/image_format"
    x_min = "image/object/bbox/xmin"
    y_min = "image/object/bbox/ymin"
    x_max = "image/object/bbox/xmax"
    y_max = "image/object/bbox/ymax"
    label_text = "image/object/class/text"
    label_id = "image/object/class/label"


IMAGE_FEATURE_MAP = {
    TFRecordLabels.width: tf.io.FixedLenFeature([], tf.int64),
    TFRecordLabels.height: tf.io.FixedLenFeature([], tf.int64),
    TFRecordLabels.filename: tf.io.FixedLenFeature([], tf.string),
    TFRecordLabels.source_id: tf.io.FixedLenFeature([], tf.string),
    TFRecordLabels.encoded: tf.io.FixedLenFeature([], tf.string),
    TFRecordLabels.file_format: tf.io.FixedLenFeature([], tf.string),
    TFRecordLabels.x_min: tf.io.VarLenFeature(tf.float32),
    TFRecordLabels.y_min: tf.io.VarLenFeature(tf.float32),
    TFRecordLabels.x_max: tf.io.VarLenFeature(tf.float32),
    TFRecordLabels.y_max: tf.io.VarLenFeature(tf.float32),
    TFRecordLabels.label_text: tf.io.VarLenFeature(tf.string),
    TFRecordLabels.label_id: tf.io.VarLenFeature(tf.int64),
}


def tfrecord_row_decode(dataset, class_table, size):
    x = tf.io.parse_single_example(dataset, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x[TFRecordLabels.encoded], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    #class_text = tf.sparse.to_dense(x[TFRecordLabels.label_text], default_value="")
    label_ids = tf.sparse.to_dense(x[TFRecordLabels.label_id])
    label_ids = tf.cast(label_ids, tf.float32)
    y_train = tf.stack(
        [
            tf.sparse.to_dense(x[TFRecordLabels.x_min]),
            tf.sparse.to_dense(x[TFRecordLabels.y_min]),
            tf.sparse.to_dense(x[TFRecordLabels.x_max]),
            tf.sparse.to_dense(x[TFRecordLabels.y_max]),
            label_ids,
            #tf.sparse.to_dense(x[TFRecordLabels.label_text]),
        ],
        axis=1,
    )

    paddings = [[0, YOLO_MAX_BOXES - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train
