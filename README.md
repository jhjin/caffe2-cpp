# Caffe2 C++ API example

The repository provides a basic image classification and detection example using Caffe2 shared library (.so).
Tested on the Ubuntu 16.04 machine.


## Dependencies

Install dependent packages via apt-get.

```bash
make dependency
make cudnn       # (optional) download publicly released cudnn v5
```

Download `cudnn` library and decompress under the `./cuda` directory which will be used by Caffe2 library.
For some reasons if you do not have an access to the latest cudnn, just proceed with `make cudnn`.
It will download publicly available cudnn (v5) - quite ancient but still ok to experiment with.

```bash
./cuda/include/cudnn.h
./cuda/lib64/libcudnn.so.*
```

Before build Caffe2, update [CMakeLists.txt](CMakeLists.txt) according to your configuration if needed.


## Build and run

The default make target will do all jobs for you - build caffe2 library, download a pretrained model (Squeeznet) and test images, compile and run the app.
The example app is heavily based on [Leo Vandriel's work](https://github.com/leonardvandriel/caffe2_cpp_tutorial/blob/master/src/caffe2/binaries/pretrained.cc).

```bash
make     # all at once
```

Or if you want to move slowly step-by-step

```bash
make model    # download model
make build    # build caffe2
make app      # compile app
make cnn      # run classification app (inference)
make rcnn     # run detection app (inference)
```

Desired outcome from the command is:

```text
$ make cnn
==> init network
==> parse image list
==> prepare batch
==> feedforward
==> retrieve results
P( lemon | lemon.jpg ) = 0.949022
P( daisy | flower.jpg ) = 0.960899
```

```text
$ make rcnn
==> using CPU
==> init network
==> parse image list
==> prepare batch (1 x 3 x 600 x 1116)
==> feedforward
==> retrieve results
P( person | street.jpg [185,186,85,260] ) = 0.999836
P( person | street.jpg [366,208,124,281] ) = 0.99977
P( person | street.jpg [754,191,108,178] ) = 0.992861
P( person | street.jpg [466,241,98,81] ) = 0.973806
P( person | street.jpg [914,180,88,219] ) = 0.95949
P( person | street.jpg [840,179,45,81] ) = 0.898059
P( person | street.jpg [921,177,82,72] ) = 0.849532
P( person | street.jpg [953,178,50,57] ) = 0.747317
P( person | street.jpg [464,245,107,186] ) = 0.702847
P( car | street.jpg [939,164,173,333] ) = 0.988054
P( car | street.jpg [556,198,30,24] ) = 0.834219
P( car | street.jpg [585,203,26,21] ) = 0.787161
P( car | street.jpg [515,199,37,24] ) = 0.756677
P( motorcycle | street.jpg [771,299,83,134] ) = 0.994598
P( truck | street.jpg [828,153,187,230] ) = 0.909398
P( truck | street.jpg [258,181,78,102] ) = 0.771324
P( couch | street.jpg [442,310,175,186] ) = 0.985259
```
