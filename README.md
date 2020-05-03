## About

Adapt the [yolov3_tf2](https://github.com/zzh8829/yolov3-tf2) into a manegable dependency


## Setup

### Creating TFRecords
installation
```bash
# for creating tfrecords and visualizing record data
pip install .[training]
```

building tfrecord dictionary
```python
from bopflow.transform.records import PascalVocDecoder


# construct tfrecord
annotation_tree = self.load_annotation(s3_annotation)
(
    s3_filepath_key,
    width,
    height,
    tf_dict
) = PascalVocDecoder.extract_image_details(annotation_tree)
tf_dict.update(
    PascalVocDecoder.extract_image_objects(
        annotation_root=annotation_tree,
        img_width=width,
        img_height=height,
        label_lookup={self.label_name: class_id},
    )
)
# load image as byte stream so to add to tf record row (tf_dict)
image_bytes = self.lake.load_picture_bytes(s3_filepath_key)
tf_dict.update(
    PascalVocDecoder.get_image_feature(image_bytes)
)

# create train object for single image
tf_object = tf.train.Example(features=tf.train.Features(feature=tf_dict))

# create writer and produce tfrecord file
writter = tf.io.TFRecordWriter(f"{self.label_name}.tfrecord")
writter.write(tf_object.SerializeToString())
writter.close()
```


### TestCommand
```bash
python bin/detect.py -image "test.jpg" --weights-path ./checkpoints/2020.04.10/yolov3.tf
```

```bash
python bin/convert.py -input checkpoints/2020.04.10/yolov3.tf --output-format model
```

```bash
python bin/visualize_dataset.py -tfrecord crosswalks.tfrecord -classes-file classes.names
```


## TODO
- [ ] verify the TF record cross walks was created correctly
  - [ ] just make sure you can extract a single image and draw/label it
- [ ] fix the training code doesn't break
- [ ] train tfrecord against it
- [ ] generate testing image
