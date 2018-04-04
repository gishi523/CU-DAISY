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

## Author
gishi523