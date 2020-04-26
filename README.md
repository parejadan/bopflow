## About

Adapt the [yolov3_tf2](https://github.com/zzh8829/yolov3-tf2) into a manegable dependency

## TestCommand
```bash
python bin/detect.py -image "test.jpg" --weights-path ./checkpoints/2020.04.10/yolov3.tf
```

```bash
python bin/convert.py -input checkpoints/2020.04.10/yolov3.tf --output-format model
```
