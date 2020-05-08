import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

from bopflow.const import YOLO_MAX_BOXES


class FeatureEncode:
    @classmethod
    def int64_list(cls, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    @classmethod
    def int64(cls, values):
        return FeatureEncode.int64_list([values])

    @classmethod
    def bytes_list(cls, values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    @classmethod
    def bytes(cls, values):
        return FeatureEncode.bytes_list([values])

    @classmethod
    def float_list(cls, values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))


class TFRRowLabels:
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


class PascalVocDecoder:
    @classmethod
    def extract_image_details(cls, annotation_root: ET.ElementTree):
        """
        Parameters
        ==========
        @annotation_root: XML with the bellow minimum structure (annotation is root)
            <annotation>
                <filename>image/filepath/cat.png</filename>
                <size>
                    <width>123</width>
                    <height>123</height>
                </size>
            </annotation>

        Returns
        =======
        (
            filename,  # str where image file is stored
            width,  # int value of image width
            height,  # int value of image height
            {
                TFRRowLabels.height: FeatureEncode.int64(height),
                TFRRowLabels.width: FeatureEncode.int64(width),
                TFRRowLabels.filename: FeatureEncode.bytes(filename.encode("UTF-8")),
                TFRRowLabels.source_id: FeatureEncode.bytes(source_id.encode("UTF-8")),
                TFRRowLabels.file_format: FeatureEncode.bytes(image_format.encode("UTF-8")),
            }
        )
        """
        filename = source_id = annotation_root.find("filename").text
        size_subtree = annotation_root.find("size")
        width = int(size_subtree.find("width").text)
        height = int(size_subtree.find("height").text)
        image_format = filename.split(".")[1]

        tf_dict = {
            TFRRowLabels.height: FeatureEncode.int64(height),
            TFRRowLabels.width: FeatureEncode.int64(width),
            TFRRowLabels.filename: FeatureEncode.bytes(filename.encode("UTF-8")),
            TFRRowLabels.source_id: FeatureEncode.bytes(source_id.encode("UTF-8")),
            TFRRowLabels.file_format: FeatureEncode.bytes(image_format.encode("UTF-8")),
        }

        return filename, width, height, tf_dict

    @classmethod
    def extract_image_objects(cls, annotation_root: ET.ElementTree):
        """
        Parameters
        ==========
        @annotation_root: XML with the bellow minimum structure (annotation is root)
            <annotation>
                <object>
                    <name>cat</name>
                    <bndbox>
                        <xmin>2</xmin>
                        <ymin>0</ymin>
                        <xmax>88</xmax>
                        <ymax>108</ymax>
                    </bndbox>
                </object>
            </annotation>

        Returns
        =======
        (
            x_mins,  # [int]
            y_mins,  # [int]
            x_maxs,  # [int]
            y_maxs,  # [int]
            label_texts,  # [str]
        )
        """
        x_mins, y_mins = [], []
        x_maxs, y_maxs = [], []
        label_texts = []
        for object_elem in annotation_root.findall("object"):
            label_texts.append(object_elem.find("name").text.encode("UTF-8"))
            bnd_box = object_elem.find("bndbox")
            # extract coordinates
            x_mins.append(int(bnd_box.find("xmin").text))
            y_mins.append(int(bnd_box.find("ymin").text))
            x_maxs.append(int(bnd_box.find("xmax").text))
            y_maxs.append(int(bnd_box.find("ymax").text))

        return x_mins, y_mins, x_maxs, y_maxs, label_texts

    @classmethod
    def normalize_text_labels(cls, label_lookup: dict, label_texts: [str]):
        """
        For the given ordered label_texts we create an equally ordered array
        representing their normalized representation. We use label_lookup
        to determine the normalization value for each label.

        Parameters
        ==========
        @label_lookup: {
            "<unique_label_text>": int,  # unique number representing label
            ...
            "<unique_label_text>": int,  # unique number representing label
        }
        @label_texts: ordered list representaing text labeling an object

        Returns
        =======
        [
            int ==> label_lookup[label_texts[index]],
            ...
        ]
        """
        label_ids = []
        for label in label_texts:
            label = label.lower()
            if isinstance(label, bytes):
                label = label.decode("UTF-8")
            try:
                normalized_id = label_lookup[label]
            except KeyError:
                raise Exception(
                    f"key {label} does not exist in label_lookup keys: {label_lookup.keys()}"
                )
            label_ids.append(normalized_id)

        return label_ids

    @classmethod
    def extract_image_objects_normalized(
        cls,
        annotation_root: ET.ElementTree,
        img_width: int,
        img_height: int,
        label_lookup: dict,
    ):
        """
        Parameters
        ==========
        @annotation_root: XML with the bellow minimum structure (annotation is root)
            <annotation>
                <filename>image/filepath/cat.png</filename>
                <size>
                    <width>123</width>
                    <height>123</height>
                </size>
            </annotation>
        @img_width: image width
        @img_height: image height
        @label_lookup: {
            "<unique_label_text>": int,  # unique number representing label
            ...
            "<unique_label_text>": int,  # unique number representing label
        }

        Returns
        =======
        {
            TFRRowLabels.x_min: FeatureEncode.float_list(x_mins),
            TFRRowLabels.y_min: FeatureEncode.float_list(y_mins),
            TFRRowLabels.x_max: FeatureEncode.float_list(x_maxs),
            TFRRowLabels.y_max: FeatureEncode.float_list(y_maxs),
            TFRRowLabels.label_text: FeatureEncode.bytes_list(label_texts),
            TFRRowLabels.label_id: FeatureEncode.int64_list(label_ids),
        }
        """
        (x_mins, y_mins, x_maxs, y_maxs, label_texts) = cls.extract_image_objects(
            annotation_root
        )
        # normalize coordinates under image dimensions so to enable reshaping
        x_mins, x_maxs = np.array(x_mins) / img_width, np.array(x_maxs) / img_width
        y_mins, y_maxs = np.array(y_mins) / img_height, np.array(y_maxs) / img_height
        label_ids = cls.normalize_text_labels(label_lookup, label_texts)

        return {
            TFRRowLabels.x_min: FeatureEncode.float_list(x_mins.tolist()),
            TFRRowLabels.y_min: FeatureEncode.float_list(y_mins.tolist()),
            TFRRowLabels.x_max: FeatureEncode.float_list(x_maxs.tolist()),
            TFRRowLabels.y_max: FeatureEncode.float_list(y_maxs.tolist()),
            TFRRowLabels.label_text: FeatureEncode.bytes_list(label_texts),
            TFRRowLabels.label_id: FeatureEncode.int64_list(label_ids),
        }

    @classmethod
    def get_image_feature(cls, image_bytes):
        return {TFRRowLabels.encoded: FeatureEncode.bytes(image_bytes)}


IMAGE_FEATURE_MAP = {
    TFRRowLabels.width: tf.io.FixedLenFeature([], tf.int64),
    TFRRowLabels.height: tf.io.FixedLenFeature([], tf.int64),
    TFRRowLabels.filename: tf.io.FixedLenFeature([], tf.string),
    TFRRowLabels.source_id: tf.io.FixedLenFeature([], tf.string),
    TFRRowLabels.encoded: tf.io.FixedLenFeature([], tf.string),
    TFRRowLabels.file_format: tf.io.FixedLenFeature([], tf.string),
    TFRRowLabels.x_min: tf.io.VarLenFeature(tf.float32),
    TFRRowLabels.y_min: tf.io.VarLenFeature(tf.float32),
    TFRRowLabels.x_max: tf.io.VarLenFeature(tf.float32),
    TFRRowLabels.y_max: tf.io.VarLenFeature(tf.float32),
    TFRRowLabels.label_text: tf.io.VarLenFeature(tf.string),
    TFRRowLabels.label_id: tf.io.VarLenFeature(tf.int64),
}


def tfrecord_row_decode(dataset, size):
    x = tf.io.parse_single_example(dataset, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x[TFRRowLabels.encoded], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    label_ids = tf.sparse.to_dense(x[TFRRowLabels.label_id])
    label_ids = tf.cast(label_ids, tf.float32)
    y_train = tf.stack(
        [
            tf.sparse.to_dense(x[TFRRowLabels.x_min]),
            tf.sparse.to_dense(x[TFRRowLabels.y_min]),
            tf.sparse.to_dense(x[TFRRowLabels.x_max]),
            tf.sparse.to_dense(x[TFRRowLabels.y_max]),
            label_ids,
        ],
        axis=1,
    )

    paddings = [[0, YOLO_MAX_BOXES - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train
