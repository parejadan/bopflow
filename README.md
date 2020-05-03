## About

Adapt the [yolov3_tf2](https://github.com/zzh8829/yolov3-tf2) into a manegable dependency

## TestCommand
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
