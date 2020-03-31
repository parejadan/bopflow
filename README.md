## About

Adapt the stable [yolov3_tf2](https://github.com/zzh8829/yolov3-tf2) into a manegable pip dependency

## TestCommand
```bash
python bin/detect.py  -image "/Users/danielpareja/sandbox/bop-projects/boplabel/data/test/cars/2d66e1c7-cb78-498d-8f40-196a49d6f923.png"
```

## Format Cleanup
```bash
docker run -v $(pwd):/code mercutiodesign/docker-black black .
```

## TODO - core code

1. [x] refactor code to core modules
   - [x] models.py
   - [x] dataset.py (parsing)
   - [x] utils.py (subsections)
   - [x] batch_norm.py
   - [x] detect.py -> bin/detect.py
   - [x] tools -> bin/*
   - [x] detect_video.py -> bin/detect_video.py
   - [x] train.py -> bin/train.py
   - [x] data + checkpoint directory (output paths)
2. [x] convert setup.py to pip dependency
3. [ ] remove absl dependency
4. [ ] strip tutorial code into documented interfaces
5. [ ] revamp docs
6. [ ] setup code for creating as pip dependency
7. [ ] deploy code to private PYPI registry (AWS)
8. [ ] add automated test coverage
9. [ ] setup automatic deployment pipeline


## TODO - training

1. [ ] create repo specifically for creating iterative versions of improved trained weights
2. [ ] deploy those weights to S3
3. [ ] figure out on that repo how to optimize training speed


## TODO - detect

1. [ ] import core code
2. [ ] snag latest stable detection weights
3. [ ] deploy to lambda function via rest framework API
