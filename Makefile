OPENCV_LIB=$(shell dirname $$(find /usr/lib -name "*opencv_imgproc*" | head -n1))
OPENCV_INCLUDE=/usr/include

BUILD_DIR=build
INCLUDE = -I./$(BUILD_DIR)/include -I$(OPENCV_INCLUDE) -I/usr/local/cuda/include -Icuda/include
LIBOPTS = -L./$(BUILD_DIR)/lib -L$(OPENCV_LIB) -L/usr/local/cuda/lib64 -Lcuda/lib64
LDFLAGS := -lcudart -lcuda -lcudnn -lcurand -lcaffe2 -lcaffe2_gpu -lgflags -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc
CFLAGS = -O3 -fpic -Wall -std=c++11
CC = gcc
CXX = g++
NB_THREADS = 8

CUDNN_URL = "http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"

.PHONY : all
all : cnn rcnn

dependency :
	sudo apt-get update
	sudo apt-get install -y --no-install-recommends \
	build-essential \
	cmake \
	git \
	libgoogle-glog-dev \
	libprotobuf-dev \
	protobuf-compiler \
	python-dev \
	python-pip
	sudo pip install numpy protobuf

opencv :
	git clone https://github.com/opencv/opencv && cd opencv; \
	git checkout 4af3ca4e4d7be246a49d751a79c6392e848ac2aa; \
	mkdir build && cd build; \
	cmake .. && make all -j2

cudnn :
	@echo
	@echo "==> Download cudnn (.tgz) from NVIDIA and untar to ./"
	@echo "    or it will use the ancient version (v5)"
	@echo
	if [ ! -d cuda ]; then \
	wget $(CUDNN_URL); \
	tar -xzf cudnn-8.0-linux-x64-v5.1.tgz; \
	rm cudnn-8.0-linux-x64-v5.1.tgz; \
	fi;

build :
	if [ ! -d cuda ]; then $(MAKE) cudnn; fi;
	rm -rf $(BUILD_DIR) && mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make -j$(NB_THREADS)

model :
	mkdir -p model/classify model/detect
	wget -O model/classify/init_net.pb https://github.com/caffe2/models/raw/master/squeezenet/init_net.pb
	wget -O model/classify/predict_net.pb https://github.com/caffe2/models/raw/master/squeezenet/predict_net.pb
	# https://github.com/facebookresearch/Detectron/blob/master/configs/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml
	wget -O model/detect/init_net.pb https://github.com/caffe2/models/raw/master/detectron/e2e_faster_rcnn_R-50-C4_2x/init_net.pb
	wget -O model/detect/predict_net.pb https://github.com/caffe2/models/raw/master/detectron/e2e_faster_rcnn_R-50-C4_2x/predict_net.pb

app : $(BUILD_DIR) model
	$(CXX) app.cc $(CFLAGS) $(INCLUDE) $(LIBOPTS) -o $@ $(LDFLAGS)

cnn : app
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:./$(BUILD_DIR)/lib:./cuda/lib64 && \
	CUDA_VISIBLE_DEVICES=0 ./app \
	--init_net model/classify/init_net.pb \
	--predict_net model/classify/predict_net.pb \
	--file $$(find image/classify -name "*.jpg" | paste -s -d, -) --gpu

rcnn : app
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:./$(BUILD_DIR)/lib:./cuda/lib64 && \
	CUDA_VISIBLE_DEVICES=0 ./app --detect --size 1333 \
	--init_net model/detect/init_net.pb \
	--predict_net model/detect/predict_net.pb \
	--file $$(find image/detect -name "*.jpg" | paste -s -d, -) \
	--output ./image/result

clean :
	rm -f *.o app

purge : clean
	rm -rf $(BUILD_DIR) model cuda
