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
3. [x] remove absl dependency
4. [x] setup code for creating as pip dependency


## TODO - scale up as service
1. [ ] create django project that imports this code
2. [ ] have project interact with S3 images
3. [ ] s3 image key paths should exists on s3 per request
4. [ ] downloads images form s3
5. [ ] returns result stating confidence level, boxing area and detection class name
6. [ ] deploy this shit to kuberneties cluster

## TODO - captcha solver
1. [ ] train detector on missing images from popular captcha puzzles
2. [ ] train detector on cryptic images from captcha puzzle
3. [ ] version weights and have code reference version

## TODO - Nice to have
4. [-] strip tutorial code into documented interfaces
5. [ ] revamp docs
6. [-] deploy code to private PYPI registry (AWS)
   - not private but s3 public access
   - [ ] add private permissions if issue
7. [ ] add automated test coverage
8. [x] setup automatic deployment pipeline
   1. [ ] add github hooks for deployment trigger


## TODO - training

1. [ ] create repo specifically for creating iterative versions of improved trained weights
2. [ ] deploy those weights to S3
3. [ ] figure out on that repo how to optimize training speed


## TODO - detect

1. [ ] import core code
2. [ ] snag latest stable detection weights
3. [ ] deploy to lambda function via rest framework API
