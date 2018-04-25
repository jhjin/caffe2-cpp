#include <tuple>

#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/utils/proto_utils.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "class_name_coco.h"
#include "class_name_imagenet.h"

// https://github.com/leonardvandriel/caffe2_cpp_tutorial/blob/master/src/caffe2/binaries/pretrained.cc
CAFFE2_DEFINE_string(init_net, "model/init_net.pb", "init net");
CAFFE2_DEFINE_string(predict_net, "model/predict_net.pb", "predict net");
CAFFE2_DEFINE_string(file, "image/classify/flower.jpg", "list of images separated by comma");
CAFFE2_DEFINE_string(output, "", "visualize detection result");
CAFFE2_DEFINE_int(size, 227, "image size in pixel");
CAFFE2_DEFINE_bool(gpu, false, "use GPU");
CAFFE2_DEFINE_bool(detect, false, "run in detection mode");
CAFFE2_DEFINE_double(threshold, 0.7, "detection score threshold");

namespace caffe2 {

cv::Mat preprocess(const std::string& image_file, int* min_size, int* max_size, bool use_crop, float& scale) {
  cv::Mat image = cv::imread(image_file);
  cv::Size dst_size;
  if (min_size != nullptr) {
    dst_size.width = std::max(*min_size * image.cols / image.rows, *min_size);
    dst_size.height = std::max(*min_size * image.rows / image.cols, *min_size);
  } else if (max_size != nullptr) {
    if (image.cols > *max_size && image.rows > *max_size) {
      dst_size.width = std::min(*max_size * image.cols / image.rows, *max_size);
      dst_size.height = std::min(*max_size * image.rows / image.cols, *max_size);
    } else {
      dst_size.width = image.cols;
      dst_size.height = image.rows;
    }
  } else {
    CAFFE_ENFORCE(false);
  }
  scale = static_cast<float>(image.cols) / dst_size.width;

  cv::resize(image, image, dst_size);
  if (use_crop) {
    int size = std::min(image.cols, image.rows);
    cv::Rect crop((image.cols - size) / 2, (image.rows - size) / 2, size, size);
    image = image(crop);
  }
  image.convertTo(image, CV_32FC3, 1.0, -128);
  return image;
}

void visualize(std::string image_file, std::vector<std::tuple<std::string, float, cv::Rect>>& boxes) {
  std::string image_filename = image_file.substr(image_file.find_last_of('/') + 1);
  std::string output = FLAGS_output + "/" + image_filename;
  cv::Mat image = cv::imread(image_file);
  for (auto& b : boxes) {
    cv::Rect& r = std::get<2>(b);
    cv::rectangle(image, r, cv::Scalar(0x00, 0x00, 0xff));
    cv::putText(image, std::get<0>(b), cv::Point(r.x, r.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0x00, 0x00, 0xff));
  }
  cv::imwrite(output, image);
}

void run() {
  DeviceOption device_option;
  std::shared_ptr<CPUContext> ctx_cpu;
  std::shared_ptr<CUDAContext> ctx_cuda;
  if (FLAGS_gpu) {
    device_option.set_device_type(CUDA);
    ctx_cuda.reset(new CUDAContext(device_option));
    std::cout << std::endl << "==> using CUDA" << std::endl;
  } else {
    device_option.set_device_type(CPU);
    ctx_cpu.reset(new CPUContext(device_option));
    std::cout << std::endl << "==> using CPU" << std::endl;
  }


  std::cout << "==> init network" << std::endl;
  NetDef init_net, predict_net;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));
  if (FLAGS_gpu) {
    init_net.mutable_device_option()->set_device_type(CUDA);
    predict_net.mutable_device_option()->set_device_type(CUDA);
  }
  Workspace workspace("default");
  CAFFE_ENFORCE(workspace.RunNetOnce(init_net));


  // https://stackoverflow.com/questions/1894886/parsing-a-comma-delimited-stdstring/10861816
  std::cout << "==> parse image list" << std::endl;
  std::stringstream ss(FLAGS_file);
  std::vector<std::string> image_list;
  while(ss.good()) {
    std::string substr;
    std::getline(ss, substr, ',');
    image_list.push_back(substr);
  }
  size_t batch_size = image_list.size();
  CAFFE_ENFORCE(!FLAGS_detect || batch_size == 1);  // force single batch in spatial mode


  std::cout << "==> prepare batch ";
  const size_t channel(3);
  std::vector<float> data_batch;
  std::vector<float> info_batch;
  for (const std::string& image_file : image_list) {
    // load image
    bool use_crop = !FLAGS_detect;
    float scale;
    cv::Mat image;
    if (FLAGS_detect) {
      image = preprocess(image_file, nullptr, &FLAGS_size, /* use_crop */ false, scale);
    } else {
      image = preprocess(image_file, &FLAGS_size, nullptr, /* use_crop */  true, scale);
    }
    info_batch.push_back(image.rows);
    info_batch.push_back(image.cols);
    info_batch.push_back(scale);

    // convert NHWC to NCHW
    std::vector<cv::Mat> channels(channel);
    cv::split(image, channels);
    std::vector<float> data;
    for (cv::Mat &c : channels) {
      data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
    }
    data_batch.insert(data_batch.end(), data.begin(), data.end());
  }

  size_t height = info_batch[0];
  size_t width = info_batch[1];
  std::cout << "(" << batch_size << " x " << channel << " x " << height << " x " << width << ")" << std::endl;
  std::vector<TIndex> dims({batch_size, channel, height, width});
  TensorCPU tensor(dims, data_batch, nullptr);
  TensorCPU im_info(std::vector<TIndex>{batch_size, 3}, info_batch, nullptr);


  std::cout << "==> feedforward" << std::endl;
  if (FLAGS_gpu) {
    workspace.CreateBlob("data")->GetMutable<TensorCUDA>()->CopyFrom(tensor);
  } else {
    workspace.CreateBlob("data")->GetMutable<TensorCPU>()->CopyFrom(tensor);
  }
  if (FLAGS_detect) {
    CAFFE_ENFORCE(FLAGS_gpu == false);  // GenerateProposals unsupported on CUDA
    workspace.CreateBlob("im_info")->GetMutable<TensorCPU>()->CopyFrom(im_info);
  }
  CAFFE_ENFORCE(workspace.RunNetOnce(predict_net));


  // run in detection mode
  if (FLAGS_detect) {
    auto score_nms = workspace.GetBlob("score_nms")->Get<TensorCPU>();
    auto bbox_nms = workspace.GetBlob("bbox_nms")->Get<TensorCPU>();
    auto class_nms = workspace.GetBlob("class_nms")->Get<TensorCPU>();
    CAFFE_ENFORCE(score_nms.dims()[0] == bbox_nms.dims()[0]);
    CAFFE_ENFORCE(score_nms.dims()[0] == class_nms.dims()[0]);
    std::cout << "==> retrieve results" << std::endl;
    std::vector<std::tuple<std::string, float, cv::Rect>> boxes;
    for (size_t i = 0; i < score_nms.dims()[0]; i++) {
      float score = score_nms.data<float>()[i];
      float cls = class_nms.data<float>()[i];
      const float* b = bbox_nms.data<float>() + 4*i;
      if (score > FLAGS_threshold) {
        cv::Rect rect = cv::Rect(b[0], b[1], b[2]-b[0]+1, b[3]-b[1]+1);
        boxes.emplace_back(class_name_coco[cls], score, rect);
      }
    }
    std::string image_file = image_list[0].substr(image_list[0].find_last_of('/') + 1);
    for (auto& b : boxes) {
      std::cout << "P( " << std::get<0>(b) << " | " << image_file << " [" << std::get<2>(b).x
                << "," << std::get<2>(b).y << "," << std::get<2>(b).width << ","
                << std::get<2>(b).height << "] ) = " << std::get<1>(b) << std::endl;
    }
    if (!FLAGS_output.empty()) {
      visualize(image_list.front(), boxes);
    }

  // run in classification mode
  } else {
    auto output = (FLAGS_gpu)?
        TensorCPU(workspace.GetBlob("softmaxout")->Get<TensorCUDA>()) :
        workspace.GetBlob("softmaxout")->Get<TensorCPU>();

    std::cout << "==> retrieve results" << std::endl;
    for (size_t i = 0; i < batch_size; i++) {
      const auto&prob = output.data<float>() + i*class_name_imagenet.size();
      std::vector<float> pred(prob, prob + class_name_imagenet.size());
      auto it = std::max_element(std::begin(pred), std::end(pred));
      auto maxValue = *it;
      auto maxIndex = std::distance(std::begin(pred), it);
      std::string image_file = image_list[i].substr(image_list[i].find_last_of('/') + 1);
      std::cout << "P( " << class_name_imagenet[maxIndex] << " | " << image_file
                << " ) = " << maxValue << std::endl;
    }
  }
}

}

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  return 0;
}
