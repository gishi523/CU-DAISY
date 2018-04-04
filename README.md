# CU-DAISY
A CUDA implementation of DAISY descriptor

## Description
- A CUDA implementation of DAISY descriptor based on [1,2,3].

## References
- [1] E. Tola, V. Lepetit, and P. Fua. Daisy: an Efficient Dense Descriptor Applied to Wide Baseline Stereo. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 32(5):815–830, 2010.
- [2] https://github.com/etola/libdaisy
- [3] https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/daisy.cpp

## Requirement
- OpenCV with cv::cuda
- CUDA
- OpenCV xfeatures2d module (optional)

## How to build
```
$ git clone https://github.com/gishi523/CU-DAISY.git
$ cd CU-DAISY
$ mkdir build
$ cd build
$ cmake ../
$ make
```

- If you want to compare with OpenCV DAISY, use
```
cmake -DWITH_OPENCV_DAISY=ON
```

## How to run
```
./cudaisy image
```

### Example
 ```
./cudaisy ./images/leuven/img1.png
```

## Performance
- Setup
  - CPU             : OpenCV DAISY with Core-i7 6700K(4.00 GHz/4Core/8T)
  - GPU             : CU-DAISY with GeForce GTX 1080
  - Image size      : 900 x 600 pixel
  - Descriptor size : 200

Parameters|CPU[msec]|GPU[msec]|Speed Up
---|---|---|---
Normalize:FULL, Interpolation:OFF|178.5|8.1|22.2
Normalize:FULL, Interpolation:ON|234.2|9.0|26.0
Normalize:PARTIAL, Interpolation:OFF|172.7|8.1|21.5
Normalize:PARTIAL, Interpolation:ON|235.0|10.0|23.5

## Author
gishi523
