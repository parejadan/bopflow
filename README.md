## About

Adapt the stable [yolov3_tf2](https://github.com/zzh8829/yolov3-tf2) into a manegable pip dependency


## TODO - core code

1. [ ] refactor code to core modules
2. [ ] strip tutorial code into documented interfaces
3. [ ] revamp docs
4. [ ] setup code for creating as pip dependency
5. [ ] deploy code to private PYPI registry (AWS)
6. [ ] add automated test coverage
7. [ ] setup automatic deployment pipeline


## TODO - training

1. [ ] create repo specifically for creating iterative versions of improved trained weights
2. [ ] deploy those weights to S3
3. [ ] figure out on that repo how to optimize training speed


## TODO - detect

1. [ ] import core code
2. [ ] snag latest stable detection weights
3. [ ] deploy to lambda function via rest framework API
