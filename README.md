## About

Adaptation of [yolov3_tf2](https://github.com/zzh8829/yolov3-tf2) into a manegable dependency


## How To

### Creating TFRecords
Training & Debug installation
```bash
# for creating tfrecords and visualizing record data
pip install .[training]
```

building tfrecord record
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


### bin Commands
```bash
python bin/convert.py -input checkpoints/2020.04.10/yolov3.tf --output-format model
```

```bash
python bin/detect.py -image "test.jpg" --weights-path ./checkpoints/2020.05.09/weights.tf
```

```bash
# requires bopflow[training] installation
python bin/visualize_dataset.py -tfrecord crosswalks-test.tfrecord && open output.png
```
![Screen Shot 2020-05-03 at 7 44 13 PM](https://user-images.githubusercontent.com/11270882/80928975-7e217280-8d76-11ea-929b-3a67de40398d.png)


```bash
# for training model
python bin/train.py \
 -tfrecord-train crosswalks-train.tfrecord \
 -tfrecord-test crosswalks-test.tfrecord \
 --weights ./checkpoints/2020.04.10/yolov3.tf \
 --epochs 10 \
 --batch-size 8 \
 --new-model-class-count 81

python bin/detect.py -image "test.jpg" --weights-path ./checkpoints/2020.05.07/yolov3_train_10.tf

```

### Format Code

```bash
docker run -v $(pwd):/code mercutiodesign/docker-black black .
```
